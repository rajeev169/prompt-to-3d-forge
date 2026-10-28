import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from bake import bake_vertex_colors


def test_bake_gradient():
    # unit quad in the z=0 plane, UVs == xy
    pos = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], float)
    uv = pos[:, :2].copy()
    faces = np.array([[0, 1, 2], [0, 2, 3]])
    g = np.linspace(0, 1, 50)
    X, Y = np.meshgrid(g, g)
    src_pos = np.stack([X.ravel(), Y.ravel(), np.zeros(X.size)], 1)
    src_col = np.stack([X.ravel(), Y.ravel(), np.zeros(X.size)], 1)  # R=x, G=y

    tex = bake_vertex_colors(uv, faces, pos, src_pos, src_col, size=64)
    assert tex.shape == (64, 64, 3)
    # row 0 is top (v=1): green high at top, low at bottom; red grows left->right
    assert tex[2, 32, 1] > 230 and tex[61, 32, 1] < 25
    assert tex[32, 61, 0] > 230 and tex[32, 2, 0] < 25


if __name__ == "__main__":
    test_bake_gradient()
    print("bake test passed")
