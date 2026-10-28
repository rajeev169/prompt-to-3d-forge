"""AI backends. Models load lazily and are reused across a batch."""
import sys
from pathlib import Path

import numpy as np

from presets import NEGATIVE

TRIPOSR_DIR = Path(__file__).parent / "third_party" / "TripoSR"


def _mesh_arrays(mesh):
    """trimesh.Trimesh -> (verts, faces, colors 0..1 or None)."""
    verts, faces = np.asarray(mesh.vertices), np.asarray(mesh.faces)
    colors = None
    try:
        vc = np.asarray(mesh.visual.vertex_colors)
        if vc.ndim == 2 and len(vc) == len(verts):
            colors = vc[:, :3].astype(np.float32) / 255.0
    except Exception:
        pass
    return verts, faces, colors


class Forge:
    def __init__(self, resolution=256, t2i_model="stable-diffusion-v1-5/stable-diffusion-v1-5"):
        import torch
        self.torch = torch
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.resolution = resolution
        self.t2i_model = t2i_model
        self._t2i = self._tsr = self._rembg = None
        self._shape = None
        print(f"[forge] device: {self.device}")

    # ---------- text -> image ----------
    def text_to_image(self, prompt, seed=0):
        torch = self.torch
        if self._t2i is None:
            from diffusers import StableDiffusionPipeline
            dtype = torch.float16 if self.device.type == "cuda" else torch.float32
            self._t2i = StableDiffusionPipeline.from_pretrained(self.t2i_model, torch_dtype=dtype).to(self.device)
        gen = torch.Generator(device="cpu").manual_seed(seed)
        return self._t2i(prompt, negative_prompt=NEGATIVE, num_inference_steps=25,
                         guidance_scale=7.5, generator=gen).images[0]

    # ---------- image -> mesh (TripoSR) ----------
    def image_to_mesh(self, image):
        torch = self.torch
        from PIL import Image
        if not TRIPOSR_DIR.exists():
            sys.exit("TripoSR not found. Run: bash setup.sh")
        sys.path.insert(0, str(TRIPOSR_DIR))
        from tsr.system import TSR
        from tsr.utils import remove_background, resize_foreground
        import rembg

        if self._tsr is None:
            self._tsr = TSR.from_pretrained("stabilityai/TripoSR", config_name="config.yaml",
                                            weight_name="model.ckpt")
            self._tsr.renderer.set_chunk_size(8192)
            self._tsr.to(self.device)
            self._rembg = rembg.new_session()

        image = remove_background(image.convert("RGB"), self._rembg)
        image = resize_foreground(image, 0.85)
        arr = np.array(image).astype(np.float32) / 255.0
        arr = arr[:, :, :3] * arr[:, :, 3:4] + (1 - arr[:, :, 3:4]) * 0.5
        image = Image.fromarray((arr * 255).astype(np.uint8))

        with torch.no_grad():
            codes = self._tsr([image], device=self.device)
        mesh = self._tsr.extract_mesh(codes, True, resolution=self.resolution)[0]
        return _mesh_arrays(mesh)

    # ---------- text -> mesh (Shap-E, lower quality, no image step) ----------
    def text_to_mesh(self, prompt, seed=0):
        torch = self.torch
        try:
            from shap_e.diffusion.sample import sample_latents
            from shap_e.diffusion.gaussian_diffusion import diffusion_from_config
            from shap_e.models.download import load_model, load_config
            from shap_e.util.notebooks import decode_latent_mesh
        except ImportError:
            sys.exit("Shap-E not installed. Run: FORGE_SHAPE=1 bash setup.sh")
        if self._shape is None:
            self._shape = (load_model("transmitter", device=self.device),
                           load_model("text300M", device=self.device),
                           diffusion_from_config(load_config("diffusion")))
        xm, model, diffusion = self._shape
        torch.manual_seed(seed)
        latents = sample_latents(
            batch_size=1, model=model, diffusion=diffusion, guidance_scale=15.0,
            model_kwargs=dict(texts=[prompt]), progress=True, clip_denoised=True,
            use_fp16=self.device.type == "cuda", use_karras=True, karras_steps=64,
            sigma_min=1e-3, sigma_max=160, s_churn=0)
        t = decode_latent_mesh(xm, latents[0]).tri_mesh()
        verts = np.asarray(t.verts, float)
        verts = np.stack([verts[:, 0], verts[:, 2], -verts[:, 1]], 1)  # Z-up -> Y-up
        colors = None
        ch = getattr(t, "vertex_channels", None)
        if ch and all(c in ch for c in "RGB"):
            colors = np.stack([np.asarray(ch[c]) for c in "RGB"], 1).astype(np.float32)
        return verts, np.asarray(t.faces), colors
