"""Mesh cleanup, normalization, decimation, UV unwrap + texture bake, GLB export."""
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from bake import bake_vertex_colors


def _trimesh():
    import trimesh
    return trimesh


def clean(mesh):
    """Merge verts, drop degenerate faces, drop tiny floating fragments."""
    tm = _trimesh()
    mesh.merge_vertices()
    mesh.update_faces(mesh.nondegenerate_faces())
    mesh.remove_unreferenced_vertices()
    parts = mesh.split(only_watertight=False)
    if len(parts) > 1:
        biggest = max(len(p.faces) for p in parts)
        keep = [p for p in parts if len(p.faces) >= 0.05 * biggest]
        mesh = tm.util.concatenate(keep)
    return mesh


def normalize(mesh, size_m):
    """Y-up, origin at bottom-center, largest dimension = size_m meters.
    Returns (offset, scale) so other point sets can follow the same transform."""
    b = mesh.bounds
    offset = np.array([-(b[0][0] + b[1][0]) / 2, -b[0][1], -(b[0][2] + b[1][2]) / 2])
    scale = size_m / float((b[1] - b[0]).max())
    mesh.apply_translation(offset)
    mesh.apply_scale(scale)
    return offset, scale


def decimate(mesh, target_faces):
    if len(mesh.faces) <= target_faces:
        return mesh.copy()
    try:
        return mesh.simplify_quadric_decimation(face_count=target_faces)
    except TypeError:  # very old trimesh
        return mesh.simplify_quadric_decimation(target_faces)


def transfer_colors(src_pos, src_col, dst_pos):
    _, nn = cKDTree(src_pos).query(dst_pos)
    return src_col[nn]


def unwrap(verts, faces):
    import xatlas
    vmap, idx, uv = xatlas.parametrize(verts.astype(np.float32), faces.astype(np.uint32))
    return verts[vmap], idx.astype(np.int64), uv


def export_vertex_color_glb(path, verts, faces, colors):
    tm = _trimesh()
    rgba = np.concatenate([(np.clip(colors, 0, 1) * 255).astype(np.uint8),
                           np.full((len(colors), 1), 255, np.uint8)], 1)
    tm.Trimesh(verts, faces, vertex_colors=rgba, process=False).export(path)


def export_textured_glb(path, verts, faces, uv, tex):
    tm = _trimesh()
    from PIL import Image
    mat = tm.visual.material.PBRMaterial(
        baseColorTexture=Image.fromarray(tex), metallicFactor=0.0, roughnessFactor=0.9)
    visual = tm.visual.TextureVisuals(uv=uv, material=mat)
    tm.Trimesh(verts, faces, visual=visual, process=False).export(path)


def build_asset(verts, faces, colors, preset, outdir, name, tex_size=1024, rotate_x=0.0):
    """Raw AI mesh -> <name>_high.glb (vertex colors) + <name>_LOD0..N.glb (UV + texture)."""
    tm = _trimesh()
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    mesh = tm.Trimesh(np.asarray(verts, float), np.asarray(faces), process=False)
    if rotate_x:
        mesh.apply_transform(tm.transformations.rotation_matrix(np.radians(rotate_x), [1, 0, 0]))
    src_pos = mesh.vertices.copy()
    src_col = (np.asarray(colors, np.float32) if colors is not None
               else np.full((len(src_pos), 3), 0.7, np.float32))

    mesh = clean(mesh)
    offset, scale = normalize(mesh, preset["size_m"])
    src_pos = (src_pos + offset) * scale

    stats = {"raw_triangles": int(len(faces)), "size_m": preset["size_m"], "files": {}}

    high = decimate(mesh, preset["high"])
    hv, hf = np.asarray(high.vertices), np.asarray(high.faces)
    hp = outdir / f"{name}_high.glb"
    export_vertex_color_glb(hp, hv, hf, transfer_colors(src_pos, src_col, hv))
    stats["files"][hp.name] = int(len(hf))

    for i, target in enumerate(preset["lods"]):
        lod = decimate(high, target)
        v, f, uv = unwrap(np.asarray(lod.vertices), np.asarray(lod.faces))
        tex = bake_vertex_colors(uv, f, v, src_pos, src_col, size=tex_size)
        lp = outdir / f"{name}_LOD{i}.glb"
        export_textured_glb(lp, v, f, uv, tex)
        stats["files"][lp.name] = int(len(f))
        print(f"  LOD{i}: {len(f)} tris")
    return stats


def write_manifest(outdir, info):
    (Path(outdir) / "manifest.json").write_text(json.dumps(info, indent=2))
