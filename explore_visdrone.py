import cv2
import os
import matplotlib
# Forces matplotlib to work on the DGX without a display
matplotlib.use('Agg') 
import matplotlib.pyplot as plt

# Updated paths for the current environment
DATA_ROOT = "data/VisDrone_Raw/VisDrone_Dataset/VisDrone2019-DET-train"
IMG_DIR = os.path.join(DATA_ROOT, "images")
LBL_DIR = os.path.join(DATA_ROOT, "annotations") 

# Category Mapping
categories = {
    1: 'pedestrian', 2: 'person', 3: 'bicycle', 4: 'car', 
    5: 'van', 6: 'truck', 7: 'tricycle', 8: 'awning-tricycle', 9: 'bus', 10: 'motor'
}

print("Starting VisDrone inspection...")

# 1. Pick an image
try:
    sample_img_name = os.listdir(IMG_DIR)[0] 
    img_path = os.path.join(IMG_DIR, sample_img_name)
    lbl_path = os.path.join(LBL_DIR, sample_img_name.replace('.jpg', '.txt'))

    # 2. Load Image
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 3. Read Annotations
    if os.path.exists(lbl_path):
        with open(lbl_path, 'r') as f:
            for line in f.readlines():
                data = line.strip().split(',')
                if len(data) < 6: continue
                
                left, top, width, height = int(data[0]), int(data[1]), int(data[2]), int(data[3])
                cat_id = int(data[5])
                
                if cat_id in categories:
                    label = categories[cat_id]
                    cv2.rectangle(image, (left, top), (left+width, top+height), (0, 255, 0), 2)
                    cv2.putText(image, label, (left, top-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

    # 4. Save visualization
    plt.figure(figsize=(12, 8))
    plt.imshow(image)
    plt.axis('off')
    plt.savefig("visdrone_inspection.png")
    print("Success: File saved as visdrone_inspection.png")

except IndexError:
    print("Error: No images found in the directory. Ensure the path is correct.")
except Exception as e:
    print(f"An error occurred: {e}")
