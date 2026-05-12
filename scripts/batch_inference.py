import os
import csv
import random
from scout_inference import AeroVLA_Engine
import config

def run_batch():
    # 1. Initialize the Engine (The logic we just tested!)
    engine = AeroVLA_Engine()
    
    # 2. Prepare the CSV file
    csv_path = os.path.join(config.OUTPUT_DIR, "final_results.csv")
    
    print(f"\n STARTING BATCH MISSION: {len(config.CATEGORIES)} categories...")
    
    with open(csv_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Category", "Image Name", "Report"])

        # 3. Loop through each category defined in config.py
        for cat in config.CATEGORIES:
            cat_path = os.path.join(config.DATASET_ROOT, cat)
            
            if not os.path.exists(cat_path):
                print(f"Skipping {cat}: Folder not found.")
                continue

            # Collect all images in subfolders
            all_images = []
            for root, dirs, files in os.walk(cat_path):
                for f in files:
                    if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                        all_images.append(os.path.join(root, f))

            # Pick 15 random images (or all if less than 15)
            selection = random.sample(all_images, min(len(all_images), config.IMAGES_PER_FOLDER))
            
            print(f"\n FOLDER: {cat.upper()} ({len(selection)} images selected)")

            for i, img_path in enumerate(selection):
                img_name = os.path.basename(img_path)
                print(f"  [{i+1}/{len(selection)}] Processing {img_name}...")
                
                # CALL THE ENGINE
                report = engine.generate_report(img_path)
                
                # Save to CSV
                writer.writerow([cat, img_name, report])

    print(f"\n MISSION COMPLETE! All reports saved to: {csv_path}")

if __name__ == "__main__":
    run_batch()
