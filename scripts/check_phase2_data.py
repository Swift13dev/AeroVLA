from data_loader import CrisisDataset
from transformers import AutoProcessor
import os

def test_loader():
    print("Initializing Phase 2 Data Loader Test...")
    
    # 1. Setup paths and processor
    tsv_path = os.path.expanduser("~/AeroVLA/data/CrisisMMD/task_humanitarian_text_img_train.tsv")
    img_root = os.path.expanduser("~/AeroVLA/data/CrisisMMD/")
    processor = AutoProcessor.from_pretrained("google/siglip-base-patch16-224")
    
    # 2. Initialize the dataset with ALL required arguments
    dataset = CrisisDataset(
        tsv_path=tsv_path,
        img_root=img_root,
        processor=processor
    )
    
    # 3. Grab the first sample
    # Note: data_loader returns (image, label, instruction)
    image, label, instruction = dataset[0]
    
    print("\n" + "="*30)
    print("EROVLA DATA VERIFICATION")
    print("="*30)
    print(f"Sample Index: 0")
    print(f"Action Command: {instruction}")
    print(f"Image Tensor Shape: {image.shape}")
    print("="*30)
    
    if "[" in instruction and "]" in instruction:
        print("\nSUCCESS: Phase 2 Action Mapping is active!")
    else:
        print("\nERROR: Action tokens not found. Check ACTION_MAP.")

if __name__ == "__main__":
    test_loader()
