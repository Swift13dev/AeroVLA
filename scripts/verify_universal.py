import torch
import random
from data_loader import UniversalCrisisDataset
from transformers import AutoProcessor, AutoModelForCausalLM
from model_bridge import AeroVLA_Bridge as AeroVLABridge

# 1. Setup
device = "cpu" # We use CPU just to test the logic while waiting
processor = AutoProcessor.from_pretrained("google/siglip-base-patch16-224")
dataset = UniversalCrisisDataset(root_dir="~/AeroVLA/data/CrisisMMD", processor=processor)

# 2. Pick 15 random indices
test_indices = random.sample(range(len(dataset)), 15)

print("\n--- AEROVLA UNIVERSAL TEST LOG ---")
for idx in test_indices:
    _, img_path = dataset[idx]
    # In the real run, we would call the model here
    print(f"TARGET IMAGE: {img_path}")
    print(f"PLAN: Generate autonomous report for this {img_path.split('/')[-2]} scene.")
    print("-" * 30)
