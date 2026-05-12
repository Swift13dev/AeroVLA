import torch
import os
import traceback # Added for deep debugging
from PIL import Image
from data_loader import UniversalCrisisDataset
from transformers import AutoProcessor, AutoModel, AutoModelForCausalLM
from model_bridge import AeroVLA_Bridge

# 1. Environment Setup
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🚀 Launching AeroVLA on {device}...")

# 2. Load Models
processor = AutoProcessor.from_pretrained("google/siglip-base-patch16-224")
vision_model = AutoModel.from_pretrained("google/siglip-base-patch16-224").vision_model.to(device)
model_brain = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM2-135M-Instruct").to(device)
bridge = AeroVLA_Bridge().to(device)

# 3. Load Universal Weights [cite: 1286]
checkpoint_path = os.path.expanduser("~/AeroVLA/models/universal_bridge.pt")
if os.path.exists(checkpoint_path):
    bridge.load_state_dict(torch.load(checkpoint_path, map_location=device))
    print("✅ Universal Weights Loaded.")
else:
    print("⚠️ Weights not found - using initial state.")

# 4. Load Universal Dataset
dataset = UniversalCrisisDataset(root_dir="~/AeroVLA/data/CrisisMMD", processor=processor)
categories = ['california_wildfires', 'hurricane_harvey', 'srilanka_floods']

print("\n--- AEROVLA MISSION LOG: UNIVERSAL REPORTS ---")

for cat in categories:
    print(f"\n📁 FOLDER: {cat.upper()}")
    # Filter only 5 images per folder to save memory
    cat_images = [p for p in dataset.samples if cat in p][:5] 
    for i, img_path in enumerate(cat_images):
        try:
            # Image Processing
            raw_image = Image.open(img_path).convert("RGB")
            inputs_v = processor(images=raw_image, return_tensors="pt").to(device)
            
            with torch.no_grad():
                # Extract pooled features (768)
                v_outputs = vision_model(inputs_v.pixel_values)
                # Map to Brain space (2048) [cite: 695]
                visual_context = bridge(v_outputs.pooler_output)
            
            # Simple prompt to avoid Tokenizer Piece ID errors [cite: 1152]
            prompt = "The disaster scene shows"
            inputs_t = processor.tokenizer(prompt, return_tensors="pt").to(device)
            
            # Use safe generation parameters [cite: 1174]
            out = model_brain.generate(
                inputs_t.input_ids,
                max_new_tokens=30,
                repetition_penalty=1.5,
                do_sample=False # Set to False for more stability
            )
            
            report = processor.tokenizer.decode(out[0], skip_special_tokens=True)
            print(f"[{i+1}/5] IMAGE: {os.path.basename(img_path)}")
            print(f"REPORT: {report}")
            print("-" * 30)
            
        except Exception:
            # THIS WILL PRINT THE ACTUAL ERROR TO YOUR TERMINAL
            traceback.print_exc()
