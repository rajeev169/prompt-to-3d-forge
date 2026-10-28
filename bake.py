"""Bake per-vertex colors from a dense mesh into a UV texture.

Pure numpy/scipy so it can be tested without torch/trimesh.
"""
import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree


def bake_vertex_colors(uv, faces, positions, src_pos, src_col, size=1024):
    """uv (N,2) in 0..1; faces (M,3); positions (N,3) of the target mesh;
    src_pos (K,3) and src_col (K,3, floats 0..1) from the dense source mesh.
    Returns a uint8 (size, size, 3) image (row 0 = top, v=1)."""
    px = np.asarray(uv, np.float64) * size
    pos = np.asarray(positions, np.float64)
    src_pos = np.asarray(src_pos, np.float64)
    src_col = np.asarray(src_col, np.float32)

    pix_xy, pix_pos = [], []
    for f in np.asarray(faces):
        tri = px[f]
        x0 = max(int(np.floor(tri[:, 0].min())), 0)
        x1 = min(int(np.ceil(tri[:, 0].max())), size - 1)
        y0 = max(int(np.floor(tri[:, 1].min())), 0)
        y1 = min(int(np.ceil(tri[:, 1].max())), size - 1)
        if x1 < x0 or y1 < y0:
            continue
        xs, ys = np.meshgrid(np.arange(x0, x1 + 1), np.arange(y0, y1 + 1))
        xs, ys = xs.ravel(), ys.ravel()
        P = np.stack([xs + 0.5, ys + 0.5], 1)
        a = tri[0]
        v0, v1, v2 = tri[1] - a, tri[2] - a, P - a
        d00, d01, d11 = v0 @ v0, v0 @ v1, v1 @ v1
        den = d00 * d11 - d01 * d01
        if abs(den) < 1e-12:
            continue
        d20, d21 = v2 @ v0, v2 @ v1
        v = (d11 * d20 - d01 * d21) / den
        w = (d00 * d21 - d01 * d20) / den
        u = 1.0 - v - w
        m = (u >= -1e-6) & (v >= -1e-6) & (w >= -1e-6)
        if not m.any():
            continue
        p3 = u[m, None] * pos[f[0]] + v[m, None] * pos[f[1]] + w[m, None] * pos[f[2]]
        pix_xy.append(np.stack([xs[m], ys[m]], 1))
        pix_pos.append(p3)

    tex = np.zeros((size, size, 3), np.float32)
    filled = np.zeros((size, size), bool)
    if pix_xy:
        xy = np.concatenate(pix_xy)
        p3 = np.concatenate(pix_pos)
        _, nn = cKDTree(src_pos).query(p3)
        tex[xy[:, 1], xy[:, 0]] = src_col[nn]
        filled[xy[:, 1], xy[:, 0]] = True

    # Pad island borders so mipmapping/filtering doesn't show black seams.
    if filled.any() and not filled.all():
        idx = ndimage.distance_transform_edt(~filled, return_distances=False, return_indices=True)
        tex = tex[idx[0], idx[1]]
    return (np.clip(tex, 0, 1)[::-1] * 255).astype(np.uint8)
