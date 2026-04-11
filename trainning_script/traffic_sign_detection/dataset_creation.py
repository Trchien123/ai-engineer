import os
import shutil
import yaml
from pathlib import Path
import random
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# Useful Link
EXTERNAL_DATASET = "./data/damaged-sign3/external_dataset_aus"
EXTERNAL_DATASET_EUROPE = "./data/damaged-sign3/external_dataset_europe"
CLEAN_DATASET = "./data/damaged-sign3/clean_dataset/train"
OUTPUT_DATASET = "./data/damaged-sign3/final_dataset_split_5"

# Hyperpaaram
TRAIN_RATIO = 0.7
VAL_RATIO = 0.2
TEST_RATIO = 0.1

FINAL_CLASSES = ["traffic_sign"]
TARGET_ID = 0
MIN_BBOX_SIZE = 0.05

# Function: convert label
dataset = []
def convert_label_to_memory(label_path):
    new_lines = []
    if not os.path.exists(label_path):
        return new_lines
        
    with open(label_path) as f:
        lines = f.readlines()

    for line in lines:
        parts = line.strip().split()
        if len(parts) < 5: continue
        
        # parts[3] is width, parts[4] is height (normalized)
        w = float(parts[3])
        h = float(parts[4])
        
        # --- FILTERING LOGIC ---
        if w < MIN_BBOX_SIZE or h < MIN_BBOX_SIZE:
            continue # Skip this specific bounding box
            
        bbox = parts[1:]
        new_lines.append((TARGET_ID, bbox))

    return new_lines

# ---- External dataset ----
for split in ["train", "valid", "test"]:
    image_dir = os.path.join(EXTERNAL_DATASET, split, "images")
    label_dir = os.path.join(EXTERNAL_DATASET, split, "labels")

    if not os.path.exists(image_dir):
        continue

    for img in os.listdir(image_dir):
        img_path = os.path.join(image_dir, img)
        label_path = os.path.join(label_dir, img.replace(".jpg",".txt").replace(".png",".txt"))

        if not os.path.exists(label_path):
            continue

        labels = convert_label_to_memory(label_path)

        if len(labels) == 0:
            continue

        dataset.append(("ext_" + img, img_path, labels))

for split in ["train", "valid"]:
    image_dir = os.path.join(EXTERNAL_DATASET_EUROPE, split, "images")
    label_dir = os.path.join(EXTERNAL_DATASET_EUROPE, split, "labels")

    if not os.path.exists(image_dir):
        continue

    for img in os.listdir(image_dir):
        img_path = os.path.join(image_dir, img)
        label_path = os.path.join(label_dir, img.replace(".jpg",".txt").replace(".png",".txt"))

        if not os.path.exists(label_path):
            continue

        labels = convert_label_to_memory(label_path)

        if len(labels) == 0:
            continue

        dataset.append(("ext_" + img, img_path, labels))

# ---- Clean dataset ----
image_dir = os.path.join(CLEAN_DATASET, "images")
label_dir = os.path.join(CLEAN_DATASET, "labels")

for img in os.listdir(image_dir):
    img_path = os.path.join(image_dir, img)
    label_path = os.path.join(label_dir, img.replace(".jpg",".txt").replace(".png",".txt"))

    if not os.path.exists(label_path):
        continue

    with open(label_path) as f:
        lines = f.readlines()

    labels = []
    for line in lines:
        parts = line.strip().split()
        bbox = parts[1:]
        # Force everything to class 0
        labels.append((TARGET_ID, bbox))

    dataset.append(("clean_" + img, img_path, labels))

print(f"Total samples collected: {len(dataset)}")

def count_classes(split):
    counts = defaultdict(int)
    for _, _, labels in split:
        for cls, _ in labels:
            counts[cls] += 1

    return counts

class_counts = count_classes(dataset)
print("\ Dataset Class Distribution:")
print("-" * 50)
print(f"{'Class':<12} {'Count':<10}")
print("-" * 50)
for cls_id, cls_name in enumerate(FINAL_CLASSES):
    count = class_counts.get(cls_id, 0)
    print(f"{cls_name:<12} {count:<10}")
print("-" * 50)


# Balance the dataset
random.shuffle(dataset)
n = len(dataset)
train_end = int(n * TRAIN_RATIO)
val_end = int(n * (TRAIN_RATIO + VAL_RATIO))

train_set = dataset[:train_end]
val_set = dataset[train_end:val_end]
test_set = dataset[val_end:]

print("Split sizes:", len(train_set), len(val_set), len(test_set))

def save_split(split_name, split_data):
    img_out = os.path.join(OUTPUT_DATASET, split_name, "images")
    lbl_out = os.path.join(OUTPUT_DATASET, split_name, "labels")

    os.makedirs(img_out, exist_ok=True)
    os.makedirs(lbl_out, exist_ok=True)

    for name, img_path, labels in split_data:
        shutil.copy(img_path, os.path.join(img_out, name))

        label_path = os.path.join(lbl_out, name.replace(".jpg",".txt").replace(".png",".txt"))

        with open(label_path, "w") as f:
            lines = []
            for cls, bbox in labels:
                lines.append(" ".join([str(cls)] + bbox))
            f.write("\n".join(lines))


save_split("train", train_set)
save_split("val", val_set)
save_split("test", test_set)

# STEP 5: YAML
# ---------------------------
final_yaml = {
    "path": OUTPUT_DATASET,
    "train": "train/images",
    "val": "val/images",
    "test": "test/images",
    "names": ["traffic_sign"],
    "nc": 1
}

with open(os.path.join(OUTPUT_DATASET, "data.yaml"), "w") as f:
    yaml.dump(final_yaml, f)

print("Done! Balanced dataset created.")