import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor
from data_loader import CrisisDataset
from torch.utils.data import DataLoader

# Setup
model_id = "HuggingFaceTB/SmolLM2-135M-Instruct"
siglip_id = "google/siglip-base-patch16-224"
json_path = "/dgxa_home/se23umcs046/AeroVLA/data/CrisisMMD/humanitarian/train.json"
img_dir = "/dgxa_home/se23umcs046/AeroVLA/data/CrisisMMD/"

print("📡 Loading Model on CPU (Verification Mode)...")
tokenizer = AutoTokenizer.from_pretrained(model_id)
processor = AutoProcessor.from_pretrained(siglip_id)
# REMOVED device_map="auto" to keep it on CPU
model = AutoModelForCausalLM.from_pretrained(model_id) 

dataset = CrisisDataset(json_path, img_dir, processor)
train_loader = DataLoader(dataset, batch_size=1, shuffle=True) # Batch size 1 for CPU

optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
model.train()

print("\n Starting CPU-based Loss Check...")
for i, (images, labels, instructions) in enumerate(train_loader):
    inputs = tokenizer(list(instructions), return_tensors="pt", padding=True)
    outputs = model(**inputs, labels=inputs["input_ids"])
    loss = outputs.loss
    
    print(f"Step {i} | Loss: {loss.item():.4f}")
    
    if i == 5: # Just get 5 steps to prove it works
        break
print("\n Logic Verified!")
