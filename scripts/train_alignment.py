import os
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader

from transformers import AutoProcessor, AutoModel

from data_loader import CrisisDataset
from model_bridge import AeroVLA_Bridge


def train_alignment():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n Starting AeroVLA Alignment Training on {device}\n")

    # ---------------------------------------------------
    # 1. Load SigLIP Vision Encoder
    # ---------------------------------------------------
    processor = AutoProcessor.from_pretrained(
        "google/siglip-base-patch16-224"
    )

    vision_model = AutoModel.from_pretrained(
        "google/siglip-base-patch16-224"
    ).vision_model.to(device)

    # Freeze SigLIP weights
    for param in vision_model.parameters():
        param.requires_grad = False

    vision_model.eval()

    # ---------------------------------------------------
    # 2. Load Bridge
    # ---------------------------------------------------
    bridge = AeroVLA_Bridge().to(device)

    # ---------------------------------------------------
    # 3. Dataset
    # ---------------------------------------------------
    tsv = os.path.expanduser(
        "~/AeroVLA/data/CrisisMMD/files_individual_events/california_wildfires_final_data.tsv"
    )

    img_dir = os.path.expanduser(
        "~/AeroVLA/data/CrisisMMD"
    )

    dataset = CrisisDataset(
        tsv,
        img_dir,
        processor
    )

    loader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=True
    )

    # ---------------------------------------------------
    # 4. Training Setup
    # ---------------------------------------------------
    optimizer = optim.Adam(
        bridge.parameters(),
        lr=5e-5
    )

    num_classes = len(dataset.label_to_id)

    classifier = nn.Linear(576, num_classes).to(device)

    criterion = nn.CrossEntropyLoss()

    # ---------------------------------------------------
    # 5. Training Loop
    # ---------------------------------------------------
    bridge.train()
    classifier.train()

    for epoch in range(5):

        running_loss = 0.0

        for i, batch in enumerate(loader):

            images, labels, instructions = batch

            images = images.to(device)
            labels = labels.to(device)

            # --------------------------------------------
            # Extract REAL semantic features using SigLIP
            # --------------------------------------------
            with torch.no_grad():

                vision_outputs = vision_model(images)

                vision_features = vision_outputs.pooler_output

            # --------------------------------------------
            # Bridge Projection
            # --------------------------------------------
            projected_features = bridge(
                vision_features
            )

            # --------------------------------------------
            # Classification Head
            # --------------------------------------------
            logits = classifier(
                projected_features
            )

            # --------------------------------------------
            # Loss
            # --------------------------------------------
            loss = criterion(
                logits,
                labels
            )

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            running_loss += loss.item()

            # --------------------------------------------
            # Logs
            # --------------------------------------------
            if i % 10 == 0:

                print(
                    f"Epoch [{epoch+1}/5] "
                    f"Step [{i}] "
                    f"Loss: {loss.item():.4f}"
                )

        avg_loss = running_loss / len(loader)

        print(
            f"\n Epoch {epoch+1} Complete "
            f"| Average Loss: {avg_loss:.4f}\n"
        )

    # ---------------------------------------------------
    # 6. Save Bridge
    # ---------------------------------------------------
    os.makedirs(
        os.path.expanduser("~/AeroVLA/models"),
        exist_ok=True
    )

    checkpoint_path = os.path.expanduser(
        "~/AeroVLA/models/universal_bridge.pt"
    )

    torch.save(
        bridge.state_dict(),
        checkpoint_path
    )

    print("\n SUCCESS: AeroVLA Bridge Training Complete")
    print(f" Saved Bridge Weights: {checkpoint_path}")


if __name__ == "__main__":
    train_alignment()
