#!/usr/bin/env python3
"""prompt-to-3d-forge: prompt (or image) -> Unity-ready GLB with LODs + baked texture."""
import argparse
import re
import sys
import time
from pathlib import Path

from presets import PRESETS, build_prompt
import mesh_tools as mt


def slug(s):
    return re.sub(r"[^a-zA-Z0-9_-]+", "_", s).strip("_")[:60] or "asset"


def parse_batch(path):
    items = []
    for n, line in enumerate(Path(path).read_text().splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = [p.strip() for p in line.split("|", 2)]
        if len(parts) != 3 or parts[1] not in PRESETS:
            sys.exit(f"{path}:{n}: expected 'name | preset | prompt' with preset in {list(PRESETS)}")
        items.append(tuple(parts))
    return items


def make_asset(forge, name, preset_key, prompt, image_path, args, out_root):
    from PIL import Image
    preset = PRESETS[preset_key]
    adir = out_root / name
    adir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    full_prompt = build_prompt(prompt, preset) if prompt else ""
    print(f"\n=== {name} [{preset_key}] via {args.backend} ===")

    if args.backend == "shap-e":
        if not prompt:
            sys.exit("Shap-E needs a text prompt.")
        verts, faces, colors = forge.text_to_mesh(full_prompt, args.seed)
    else:
        if image_path:
            image = Image.open(image_path).convert("RGBA")
        else:
            image = forge.text_to_image(full_prompt, args.seed)
            image.save(adir / "concept.png")
        verts, faces, colors = forge.image_to_mesh(image)

    print(f"  raw mesh: {len(faces)} tris")
    stats = mt.build_asset(verts, faces, colors, preset, adir, name,
                           tex_size=args.tex_size, rotate_x=args.rotate_x)
    stats.update(name=name, preset=preset_key, prompt=prompt, backend=args.backend,
                 seed=args.seed, seconds=round(time.time() - t0, 1))
    mt.write_manifest(adir, stats)
    return stats


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--prompt", help="text description of the asset")
    ap.add_argument("--image", help="your own concept image (best quality)")
    ap.add_argument("--batch", help="file with lines: name | preset | prompt")
    ap.add_argument("--name", help="asset name (single mode)")
    ap.add_argument("--preset", default="prop", choices=list(PRESETS))
    ap.add_argument("--backend", default="triposr", choices=["triposr", "shap-e"])
    ap.add_argument("--resolution", type=int, default=256, help="marching-cubes grid; higher = denser mesh (320-512 needs a GPU)")
    ap.add_argument("--tex-size", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rotate-x", type=float, default=0.0, help="fix orientation if model comes out lying down (e.g. -90)")
    ap.add_argument("--t2i-model", default="stable-diffusion-v1-5/stable-diffusion-v1-5")
    ap.add_argument("--out", default="outputs")
    args = ap.parse_args()

    if not (args.prompt or args.image or args.batch):
        ap.error("give --prompt, --image, or --batch")

    from backends import Forge
    forge = Forge(resolution=args.resolution, t2i_model=args.t2i_model)
    out_root = Path(args.out)

    if args.batch:
        jobs = [(n, p, t, None) for n, p, t in parse_batch(args.batch)]
    else:
        name = slug(args.name or (args.prompt or Path(args.image).stem))
        jobs = [(name, args.preset, args.prompt, args.image)]

    results = []
    for name, preset_key, prompt, image in jobs:
        try:
            results.append(make_asset(forge, name, preset_key, prompt, image, args, out_root))
        except Exception as e:  # keep a long batch going
            print(f"  !! {name} failed: {e}")

    print("\nDone:", len(results), "of", len(jobs), "assets ->", out_root.resolve())


if __name__ == "__main__":
    main()
