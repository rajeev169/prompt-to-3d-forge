# prompt-to-3d-forge

Prompt (or your own concept image) → AI mesh → cleaned, scaled, **LOD'd, UV-unwrapped, textured GLB** for Unity.

Per asset you get:
- `<name>_high.glb` – dense mesh with vertex colors (bake source / close-ups)
- `<name>_LOD0..2.glb` – game-ready meshes with UVs and a baked 1024px texture
- `concept.png` (text mode) and `manifest.json` (triangle counts, prompt, seed)

Triangle budgets and real-world sizes are in `presets.py` (tuned for mobile).

## Run it

**GitHub Actions (no GPU, slow but free):** Actions → *Generate 3D Game Assets* → Run workflow. Download the `game-assets-N` artifact. Use `mode: batch` to build the whole list in `assets/downhill_forest.txt`.

**Google Colab / Kaggle / local GPU (recommended for quality):**
```bash
git clone <your-repo> && cd prompt-to-3d-forge
FORGE_CUDA=1 bash setup.sh
python generate.py --name pine_tree --preset tree --prompt "tall pine tree" --resolution 384
python generate.py --batch assets/downhill_forest.txt --resolution 384
```

**Best quality trick:** pass your own clean concept image: `--image my_rock.png --name rock_a --preset rock`.

Self-hosted GPU runner: add a repo variable `RUNNER` with its label and the workflow uses it.

## Into Unity
1. Install the `UnityGLTF` (or glTFast) package and drop the `.glb` files in `Assets/`.
2. For automatic LOD Groups: `blender -b -P blender/glb_to_fbx.py -- outputs/pine_tree` makes one FBX containing `_LOD0/_LOD1/_LOD2`, which Unity turns into a LODGroup on import.
3. Set the material to URP/Lit (mobile), enable GPU instancing, and use LOD0 only near the camera.

## Honest limits
- AI image-to-3D gives blobby geometry with **no rig and no PBR maps**. Great for rocks, logs, stumps, signs, plants, trees; weak for bikes (wheels fuse into the frame) and characters. Model those in Blender.
- "High poly" detail is limited by `--resolution`; 384–512 needs a GPU.
- If a model comes out lying down, use `--rotate-x -90`.
- CPU runs take several minutes per asset.
- Not yet run end to end with the real models. Only the texture baker has a test (`python tests/test_bake.py`). Expect small fixes on first run, especially dependency versions in `setup.sh`.

## Licenses (check before shipping)
TripoSR (MIT), Shap-E (MIT), Stable Diffusion 1.5 (CreativeML OpenRAIL-M, use restrictions apply). Verify current terms on each model page.
