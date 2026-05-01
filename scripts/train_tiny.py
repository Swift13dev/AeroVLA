import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import AutoProcessor
# Importing the pieces we built yesterday
from data_loader import CrisisDataset
from model_bridge import AeroVLA_Bridge

def train():
    # 1. Setup Device (Supercomputer GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Training on: {device}")

    # 2. Initialize "Eyes" and "Bridge"
    processor = AutoProcessor.from_pretrained("google/siglip-base-patch16-224")
    model = AeroVLA_Bridge().to(device)
    
    # 3. Load Data (Testing on the California Wildfires folder)
    data_dir = os.path.expanduser("~/AeroVLA/data/CrisisMMD/data_image/california_wildfires/10_10_2017")
    dataset = CrisisDataset(data_dir, processor)
    loader = DataLoader(dataset, batch_size=2, shuffle=True)

    # 4. Optimizer (The "Adjuster")
    # We only train the Projector, not the whole brain (this is called "Frozen Backbone" training)
    optimizer = optim.Adam(model.projector.parameters(), lr=1e-4)
    criterion = nn.MSELoss()

    print("🏁 Starting Tiny Training Run...")
    
    model.train()
    for epoch in range(3): # Just 3 loops for this test
        total_loss = 0
        for batch in loader:
            batch = batch.to(device)
            
            # Forward Pass: Eyes -> Bridge
            # (In this test, we simulate the 'target' as random numbers)
            optimizer.zero_grad()
            output = model(batch.view(batch.size(0), -1)[:, :768]) 
            
            # Dummy target for alignment test
            target = torch.randn(output.size()).to(device)
            
            loss = criterion(output, target)
            
            # Backward Pass: The AI learns from its mistake
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        print(f"Epoch {epoch+1} | Loss: {total_loss/len(loader):.4f}")

    print(" SUCCESS: The Bridge has completed its first learning cycle!")

if __name__ == "__main__":
    import os
    train()
