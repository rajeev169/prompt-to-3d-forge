"""Merge an asset's LOD GLBs into one FBX that Unity turns into a LODGroup.

Usage:  blender -b -P blender/glb_to_fbx.py -- outputs/pine_tree
Meshes are named <name>_LOD0, <name>_LOD1, ... which Unity auto-detects on FBX import.
"""
import glob
import os
import re
import sys

import bpy

adir = sys.argv[sys.argv.index("--") + 1].rstrip("/")
name = os.path.basename(adir)

bpy.ops.wm.read_factory_settings(use_empty=True)
for path in sorted(glob.glob(os.path.join(adir, f"{name}_LOD*.glb"))):
    lod = re.search(r"_LOD(\d+)\.glb$", path).group(1)
    bpy.ops.import_scene.gltf(filepath=path)
    for obj in list(bpy.context.selected_objects):
        if obj.type == "MESH":
            obj.name = f"{name}_LOD{lod}"
            obj.data.name = obj.name

bpy.ops.object.select_all(action="SELECT")
bpy.ops.export_scene.fbx(
    filepath=os.path.join(adir, f"{name}.fbx"),
    use_selection=True, path_mode="COPY", embed_textures=True,
    apply_scale_options="FBX_SCALE_ALL",
)
print("Wrote", os.path.join(adir, f"{name}.fbx"))
