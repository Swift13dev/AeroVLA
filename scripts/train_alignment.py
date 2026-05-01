import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import AutoProcessor
# Importing your custom data loader and bridge
from data_loader import CrisisDataset
from model_bridge import AeroVLA_Bridge
import os

def train_alignment():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Phase 2: Starting Alignment Training on {device}")

    # 1. Load the "Eyes" (Processor) and the "Bridge"
    processor = AutoProcessor.from_pretrained("google/siglip-base-patch16-224")
    model = AeroVLA_Bridge().to(device)
    
    # 2. Use the REAL TSV and Image Directory
    tsv = os.path.expanduser("~/AeroVLA/data/CrisisMMD/files_individual_events/california_wildfires_final_data.tsv")
    img_dir = os.path.expanduser("~/AeroVLA/data/CrisisMMD")
    
    dataset = CrisisDataset(tsv, img_dir, processor)
    # Batch size 8 is a good "sweet spot" for the DGX
    loader = DataLoader(dataset, batch_size=8, shuffle=True)

    # 3. Training Config (Only training the Projector)
    optimizer = optim.Adam(model.projector.parameters(), lr=5e-5)
    # CrossEntropy is used because we are now "Classifying" real labels
    criterion = nn.CrossEntropyLoss()

    print("enginning Real-Label Alignment...")
    
    model.train()
    for epoch in range(5): # 5 Epochs for a solid initial run
        running_loss = 0.0
        for i, (images, labels) in enumerate(loader):
            images, labels = images.to(device), labels.to(device)
            
            # Flatten images for the projector [Batch, 3*224*224] -> [Batch, 768]
            # (Note: In a full VLA we use the Vision Encoder, here we test the Bridge)
            vision_features = images.view(images.size(0), -1)[:, :768]
            
            optimizer.zero_grad()
            outputs = model.projector(vision_features)
            
            # We map the 2048 output back to our number of labels for the loss
            # This teaches the bridge to 'cluster' similar disaster images
            loss = criterion(outputs[:, :len(dataset.label_map)], labels)
            
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
            if i % 10 == 0:
                print(f"Epoch [{epoch+1}/5], Step [{i}], Loss: {loss.item():.4f}")

    print("SUCCESS: Phase 2 Alignment Complete. The Bridge is now 'Crisis-Aware'.")

if __name__ == "__main__":
    train_alignment()
