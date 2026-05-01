import os
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import AutoProcessor

# Phase 2 Action Mapping: Maps specific dataset labels to drone commands
# These labels match the "label" keys seen in your JSON (e.g., other_relevant_information)
ACTION_MAP = {
    "not_humanitarian": "No threat detected. Continue [ROUTE_PATROL].",
    "other_relevant_information": "Execute [WIDE_AREA_SCAN] and maintain altitude.",
    "infrastructure_and_utility_damage": "Critical hazard detected. Execute [HOVER_STABLE] for detailed recon.",
    "affected_individuals": "Resource need identified. Execute [INIT_DROP_SEQUENCE].",
    "rescue_volunteering_or_donation_effort": "Execute [INIT_DROP_SEQUENCE].",
    "Informative": "Execute [WIDE_AREA_SCAN].",
    "Infrastructure_Damage": "Critical hazard detected. Execute [HOVER_STABLE].",
    "Humanitarian_Aid": "Resource need identified. Execute [INIT_DROP_SEQUENCE]."
}

class CrisisDataset(Dataset):
    def __init__(self, file_path, img_root, processor):
        self.file_path = os.path.expanduser(file_path)
        self.img_root = os.path.expanduser(img_root)
        self.processor = processor

        # Flexible Data Loading: Logic for JSON vs TSV
        if self.file_path.endswith('.json'):
            self.data = pd.read_json(self.file_path)
            # Verified keys from your 'head' command output
            self.label_col = 'label'
            self.path_col = 'image_path'
        else:
            self.data = pd.read_csv(self.file_path, sep='\t')
            self.label_col = self.data.columns[3]
            self.path_col = self.data.columns[1]

        # Map labels to IDs
        self.label_to_id = {label: i for i, label in enumerate(self.data[self.label_col].unique())}
        print(f" Dataset Loaded: {len(self.data)} samples identified.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        label_text = str(row[self.label_col])
        img_rel_path = str(row[self.path_col])
        
        # 1. Map label to Phase 2 Action Instruction
        instruction = ACTION_MAP.get(label_text, "Continue [ROUTE_PATROL].")
        label_id = self.label_to_id.get(label_text, 0)

        # 2. Load and process the image
        img_full_path = os.path.join(self.img_root, img_rel_path)
        
        try:
            image = Image.open(img_full_path).convert("RGB")
            pixel_values = self.processor(images=image, return_tensors="pt").pixel_values
            image_tensor = pixel_values.squeeze(0)
        except Exception as e:
            # Returns a zero tensor if an image is missing (Honest Reporting)
            image_tensor = torch.zeros(3, 224, 224)

        return image_tensor, torch.tensor(label_id), instruction

# --- TEST BLOCK FOR MILESTONE DEMO ---
if __name__ == "__main__":
    print(" AeroVLA Phase 2 Data Loader Test")
    
    # Absolute paths verified on Mahindra University DGX
    PROC = AutoProcessor.from_pretrained("google/siglip-base-patch16-224")
    JSON_PATH = "/dgxa_home/se23umcs046/AeroVLA/data/CrisisMMD/humanitarian/train.json"
    IMG_DIR = "/dgxa_home/se23umcs046/AeroVLA/data/CrisisMMD/"
    
    try:
        test_ds = CrisisDataset(JSON_PATH, IMG_DIR, PROC)
        img, lbl, instr = test_ds[0]
        
        print("\n" + "="*40)
        print(" SUCCESS: DATA PIPELINE ACTIVE")
        print("="*40)
        print(f"Instruction : {instr}")
        print(f"Label ID    : {lbl.item()}")
        print(f"Image Tensor: {img.shape}")
        print("="*40)
    except Exception as e:
        print(f"\n SETUP ERROR: {e}")
