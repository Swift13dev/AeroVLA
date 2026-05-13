import torch
import torch.nn as nn
from transformers import CLIPModel, CLIPProcessor, AutoModelForCausalLM, AutoTokenizer
from PIL import Image
import os
import random
from aerovla_bridge import AeroVLABridge
from visdrone_captioner import VisDroneCaptioner

# 1. System Configuration
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load Models and Force to Float32 for cross-platform stability
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device).float()
smol_model = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM2-135M").to(device).float()
tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Freeze Pre-trained Backbones (Only training the Bridge)
for param in clip_model.parameters():
    param.requires_grad = False
for param in smol_model.parameters():
    param.requires_grad = False

# Initialize AeroVLA Bridge
bridge = AeroVLABridge().to(device).float()
captioner = VisDroneCaptioner()
optimizer = torch.optim.Adam(bridge.parameters(), lr=1e-4)
criterion = nn.MSELoss()

# 2. Dataset Paths
DATA_ROOT = "data/VisDrone_Raw/VisDrone_Dataset/VisDrone2019-DET-train"
IMG_DIR = os.path.join(DATA_ROOT, "images")
LBL_DIR = os.path.join(DATA_ROOT, "annotations")

# FINAL PHASE: Use all 6,471 images
all_images = [f for f in os.listdir(IMG_DIR) if f.lower().endswith('.jpg')]
random.shuffle(all_images)
train_subset = all_images 

print(f"Starting Final Phase Training: {len(train_subset)} images | 5 Epochs")

# 3. Training Loop
for epoch in range(5):
    epoch_loss = 0
    processed_count = 0
    
    for i, img_name in enumerate(train_subset):
        try:
            img_path = os.path.join(IMG_DIR, img_name)
            lbl_path = os.path.join(LBL_DIR, img_name.replace('.jpg', '.txt'))
            
            # A. Visual Feature Extraction (768-D)
            image = Image.open(img_path).convert("RGB")
            inputs = processor(images=image, return_tensors="pt").to(device)
            
            with torch.no_grad():
                vision_outputs = clip_model.vision_model(pixel_values=inputs["pixel_values"])
                visual_embeds = vision_outputs.pooler_output.float()

            # B. Semantic Target Generation (576-D)
            caption = captioner.generate_caption(lbl_path)
            with torch.no_grad():
                text_inputs = tokenizer(caption, return_tensors="pt").to(device)
                target_sem = smol_model.model.embed_tokens(text_inputs["input_ids"]).mean(dim=1).float()

            # C. Bridge Optimization
            projected_output = bridge(visual_embeds)
            loss = criterion(projected_output, target_sem)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            processed_count += 1
            
            # Print progress every 1000 images for the full run
            if (i + 1) % 1000 == 0:
                print(f"  > Epoch {epoch+1} | Progress: {i+1}/{len(train_subset)} | Loss: {loss.item():.6f}")

        except Exception:
            continue
            
    # Status Update
    if processed_count > 0:
        avg_loss = epoch_loss / processed_count
        print(f"Completed Epoch {epoch+1} | Average Alignment Loss: {avg_loss:.6f}")
    else:
        print(f"Failed Epoch {epoch+1}: No images were processed. Verify dataset paths.")

# 4. Save Final Production Weights
torch.save(bridge.state_dict(), "aerovla_bridge_final.pth")
print("Final Production Training Complete. Weights saved as aerovla_bridge_final.pth")