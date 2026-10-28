import sys, torch, trimesh
from PIL import Image

model_choice = sys.argv[1].split(' - ')[0] # "TripoSR" from "TripoSR - Best free quality"
prompt = sys.argv[2]
poly_count = sys.argv[3]

print(f"Model: {model_choice} | Prompt: {prompt} | Target: {poly_count}")

# Poly targets
POLY_TARGETS = {'low': 5000, 'medium': 20000, 'high': 100000}

def save_glb(mesh, name='output.glb'):
    mesh.export(name)
    print(f"Saved {name}")

def decimate(mesh, target_faces):
    print(f"Decimating to ~{target_faces} faces...")
    mesh = mesh.simplify_quadric_decimation(face_count=target_faces)
    return mesh

if model_choice == "Shap-E":
    from shap_e.diffusion.sample import sample_latents
    from shap_e.diffusion.gaussian_diffusion import diffusion_from_config
    from shap_e.models.download import load_model, load_config
    from shap_e.util.notebooks import decode_latent_mesh
    
    device = torch.device('cpu')
    xm = load_model('transmitter', device=device)
    model = load_model('text300M', device=device)
    diffusion = diffusion_from_config(load_config('diffusion'))
    
    latents = sample_latents(
        batch_size=1, model=model, diffusion=diffusion,
        guidance_scale=15.0, model_kwargs=dict(texts=[prompt]),
        progress=True, clip_denoised=True, use_fp16=False,
        use_karras=True, karras_steps=64, sigma_min=1e-3, sigma_max=160, s_churn=0,
    )
    mesh = decode_latent_mesh(xm, latents[0]).tri_mesh()
    mesh = trimesh.Trimesh(vertices=mesh.verts, faces=mesh.faces)
    
elif model_choice == "TripoSR":
    from diffusers import StableDiffusionPipeline
    from tsr.system import TSR
    from tsr.utils import remove_background, resize_foreground
    
    # Step 1: Prompt -> Image
    print("Generating concept image...")
    pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float32)
    pipe = pipe.to("cpu")
    image = pipe(prompt + ", 3d render, white background, centered, game asset").images[0]
    image.save("concept.png")
    
    # Step 2: Image -> 3D
    print("Converting image to 3D...")
    model = TSR.from_pretrained("stabilityai/TripoSR", config_name="config.yaml", weight_name="model.ckpt")
    model.to("cpu")
    image = remove_background(image)
    image = resize_foreground(image, 0.85)
    image = image.convert("RGB")
    with torch.no_grad():
        scene_codes = model(image, device="cpu")
    mesh = model.extract_mesh(scene_codes)[0]
    
elif model_choice == "Luma Genie":
    print("Luma Genie needs API key. Set LUMA_API_KEY in repo Secrets to use.")
    print("Falling back to TripoSR for now...")
    # Just run TripoSR code again as fallback
    exec(open(__file__).read().replace('model_choice == "Luma Genie"', 'False'))
    exit()
    
else:
    raise ValueError(f"Unknown model: {model_choice}")

# Decimate to target poly count
mesh = decimate(mesh, POLY_TARGETS[poly_count])
save_glb(mesh)
print("ALL DONE - Download from Artifacts")
