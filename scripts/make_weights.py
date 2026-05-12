import torch
import os
from model_bridge import AeroVLA_Bridge

# Ensure the folder exists
os.makedirs("../models", exist_ok=True)

# Create the bridge and save its state
bridge = AeroVLA_Bridge()
save_path = "../models/universal_bridge.pt"
torch.save(bridge.state_dict(), save_path)

print(f"Emergency weights created at: {save_path}")
