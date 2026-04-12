import cv2
import os

INPUT_DIR = "data/raw"
OUTPUT_DIR = "data/processed"

os.makedirs(OUTPUT_DIR, exist_ok=True)

for file in os.listdir(INPUT_DIR):
    img_path = os.path.join(INPUT_DIR, file)
    img = cv2.imread(img_path)

    if img is None:
        continue

    img = cv2.resize(img, (256, 256))
    cv2.imwrite(os.path.join(OUTPUT_DIR, file), img)

print("Preprocessing done ✅")