import torch
from transformers import CLIPModel, CLIPProcessor, AutoModelForCausalLM, AutoTokenizer
from PIL import Image
import os
from aerovla_bridge import AeroVLABridge

# 1. System Setup
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load Models (Forced to Float32)
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device).float()
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
smol_model = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM2-135M").to(device).float()
tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M")

# Load trained Bridge
bridge = AeroVLABridge().to(device).float()
bridge.load_state_dict(torch.load("aerovla_bridge_final.pth", map_location=device))

clip_model.eval()
smol_model.eval()
bridge.eval()

# MU-Specific Semantic Vocabulary
semantic_labels = [
    "campus buildings", "open ground", "road pathway", "parking area",
    "pedestrian walkway", "sports field", "trees and vegetation",
    "urban campus layout", "vehicles parked", "outdoor environment",
    "building entrance", "recreational area"
]

def observe_aerial_scene(image_path):
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(device)
    
    with torch.no_grad():
        vision_outputs = clip_model.vision_model(pixel_values=inputs["pixel_values"])
        visual_embeds = vision_outputs.pooler_output.float()
        projected_embeds = bridge(visual_embeds).unsqueeze(1)
        
        # We REMOVE the list of labels from the prompt to stop the model from getting confused.
        # Instead, we give it a direct command.
        prompt = (
            "Analyze aerial drone image. Identify the main object.\n"
            "Observation: "
        )
        
        prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        prompt_embeds = smol_model.model.embed_tokens(prompt_ids).float()
        
        full_embeds = torch.cat([projected_embeds, prompt_embeds], dim=1)
        attention_mask = torch.ones(full_embeds.shape[:2], dtype=torch.long).to(device)
        
        output_ids = smol_model.generate(
            inputs_embeds=full_embeds,
            attention_mask=attention_mask,
            max_new_tokens=10, 
            do_sample=False, 
            repetition_penalty=2.0,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
        
        res = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return res.replace(prompt, "").strip()

# 2. MU IMAGE DIRECTORY SETUP
IMG_DIR = "MU_Validation/images/MU_Validation_Images"

print("\n--- AEROVLA MU CAMPUS INTELLIGENCE REPORT ---")

if not os.path.exists(IMG_DIR):
    print(f"ERROR: Directory '{IMG_DIR}' not found. Please check your path.")
else:
    # MULTI-FORMAT SUPPORT (.jpg, .png, .jpeg, .jfif)
    test_images = [
        f for f in os.listdir(IMG_DIR) 
        if f.lower().endswith(('.jpg', '.jpeg', '.png', '.jfif'))
    ][:5]

    print(f"Detected {len(test_images)} images for validation\n")

    if len(test_images) == 0:
        print("No valid images found. Check file extensions or upload images to the folder.")
    else:
        for img_name in test_images:
            path = os.path.join(IMG_DIR, img_name)
            print(f"Processing: {img_name}")
            try:
                prediction = observe_aerial_scene(path)
                print(f"Result: {prediction}\n")
            except Exception as e:
                print(f"Error analyzing {img_name}: {e}\n")
