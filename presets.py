"""Asset presets: size in meters, triangle budgets (mobile-friendly), prompt hints."""

NEGATIVE = (
    "blurry, low quality, multiple objects, cropped, text, watermark, logo, "
    "ground plane, cast shadow, busy background, people"
)
BASE = (
    "single object, centered, full view, 3/4 angle, plain white background, "
    "soft even lighting, highly detailed, realistic textures, game asset, 3d render"
)

# high = dense bake-source mesh. lods = triangle budgets for LOD0, LOD1, LOD2.
PRESETS = {
    "prop":      dict(size_m=1.0, high=100_000, lods=[8000, 3000, 800],   hint=""),
    "rock":      dict(size_m=1.5, high=100_000, lods=[3000, 1000, 300],   hint="natural rock, weathered"),
    "tree":      dict(size_m=8.0, high=150_000, lods=[8000, 3000, 800],   hint="full tree with trunk and foliage"),
    "plant":     dict(size_m=0.8, high=60_000,  lods=[2000, 800, 300],    hint="natural plant"),
    "character": dict(size_m=1.8, high=150_000, lods=[20000, 8000, 2500], hint="full body, T-pose"),
    "vehicle":   dict(size_m=2.0, high=150_000, lods=[20000, 8000, 2500], hint="static, side three-quarter view"),
    "building":  dict(size_m=6.0, high=150_000, lods=[15000, 6000, 2000], hint="whole structure"),
}


def build_prompt(prompt: str, preset: dict) -> str:
    parts = [prompt]
    if preset["hint"]:
        parts.append(preset["hint"])
    parts.append(BASE)
    return ", ".join(parts)
