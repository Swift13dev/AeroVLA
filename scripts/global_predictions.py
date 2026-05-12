import torch
import os
from data_loader import UniversalCrisisDataset
from transformers import AutoProcessor, AutoModelForCausalLM

# 1. Setup
device = "cpu" # Use CPU for now while Terminal 1 waits for GPU
processor = AutoProcessor.from_pretrained("google/siglip-base-patch16-224") #
model = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM2-135M-Instruct").to(device) #

# 2. Get the folders
categories = ['california_wildfires', 'hurricane_harvey', 'hurricane_irma', 
              'hurricane_maria', 'iraq_iran_earthquake', 'mexico_earthquake', 'srilanka_floods'] #

print("--- AEROVLA UNIVERSAL DISASTER ASSESSMENT LOG ---")

for cat in categories:
    print(f"\n CATEGORY: {cat.upper()}")
    # Filter dataset for only this category
    cat_images = [path for path in UniversalCrisisDataset("~/AeroVLA/data/CrisisMMD", processor).samples if cat in path]
    
    # Pick 15 samples
    selection = cat_images[:15] 
    
    for i, img_path in enumerate(selection):
        # In a real run, this is where SigLIP and SmolLM2 generate the text [cite: 199, 200]
        print(f"[{i+1}/15] IMAGE: {os.path.basename(img_path)}")
        print(f"PREDICTION: [AeroVLA is analyzing {cat} patterns...]")
