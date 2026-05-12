import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import AutoProcessor
import os # Added missing import

# Importing the pieces we built
from data_loader import CrisisDataset
from model_bridge import AeroVLA_Bridge

def train():
    # 1. Setup Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f" Training on: {device}")

    # 2. Initialize "Eyes" and "Bridge"
    processor = AutoProcessor.from_pretrained("google/siglip-base-patch16-224")
    model = AeroVLA_Bridge().to(device)

    # 3. Load Data
    data_dir = os.path.expanduser("~/AeroVLA/data/CrisisMMD/data_image/california_wildfires/10_10_2017")
    
    # FIXED INDENTATION HERE
    dataset = CrisisDataset(root_dir=data_dir, processor=processor)
    loader = DataLoader(dataset, batch_size=2, shuffle=True)

    # 4. Optimizer
    optimizer = optim.Adam(model.projector.parameters(), lr=1e-4)
    criterion = nn.MSELoss()

    print("Starting Tiny Training Run...")

    model.train()
    for epoch in range(3): 
        total_loss = 0
        for batch in loader:
            # Check if batch is a tensor or dict
            pixel_values = batch['pixel_values'].to(device) if isinstance(batch, dict) else batch.to(device)
            
            optimizer.zero_grad()
            
            # Flatten pixels to match bridge input (768)
            # We take the mean across the spatial dimensions to simulate pooler_output
            if pixel_values.ndim == 4:
                features = pixel_values.mean(dim=[2, 3])[:, :768]
            else:
                features = pixel_values.view(pixel_values.size(0), -1)[:, :768]

            output = model(features)

            # Dummy target for alignment test
            target = torch.randn(output.size()).to(device)
            loss = criterion(output, target)

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1} | Loss: {total_loss/len(loader):.4f}")

    # 5. SAVE THE WEIGHTS (Crucial for stopping hallucinations!)
    save_path = os.path.expanduser("~/AeroVLA/models/universal_bridge.pt")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f" SUCCESS: Weights saved to {save_path}")

if __name__ == "__main__":
    train()
