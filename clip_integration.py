import torch
from transformers import CLIPModel, CLIPProcessor
from PIL import Image
import os

# ---------------------------------------------------
# AeroVLA Phase 2 — CLIP Integration Test
# ---------------------------------------------------

# Device setup
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Running on: {device}")

# ---------------------------------------------------
# 1. Load CLIP
# ---------------------------------------------------

print("Loading CLIP model...")

model = CLIPModel.from_pretrained(
    "openai/clip-vit-base-patch32"
).to(device)

processor = CLIPProcessor.from_pretrained(
    "openai/clip-vit-base-patch32"
)

# ---------------------------------------------------
# 2. Load Sample VisDrone Image
# ---------------------------------------------------

IMG_DIR = "data/VisDrone_Raw/VisDrone_Dataset/VisDrone2019-DET-train/images"

# Pick first image
sample_img = os.path.join(
    IMG_DIR,
    os.listdir(IMG_DIR)[0]
)

print(f"Image selected: {os.path.basename(sample_img)}")

# Open image
image = Image.open(sample_img).convert("RGB")

# ---------------------------------------------------
# 3. Preprocess Image
# ---------------------------------------------------

inputs = processor(
    images=image,
    return_tensors="pt"
)

# Move only tensor values to device
inputs = {
    k: v.to(device)
    for k, v in inputs.items()
}

# ---------------------------------------------------
# 4. Extract CLIP Visual Embeddings
# ---------------------------------------------------

with torch.no_grad():

    outputs = model.vision_model(
        pixel_values=inputs["pixel_values"]
    )

    # Extract pooled CLIP features
    embeddings = outputs.pooler_output

# ---------------------------------------------------
# 5. Output Verification
# ---------------------------------------------------

print("\n--- CLIP Integration Success ---")

print(f"Processed Image : {os.path.basename(sample_img)}")

print(f"Embedding Shape : {embeddings.shape}")

print(f"Embedding DType : {embeddings.dtype}")

print("\nSample Embedding Values:")
print(embeddings[0][:10])
