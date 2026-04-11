import os
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import random

OUTPUT_DATASET = "./data/damaged-sign3/final_dataset_split_5"
CLEAN_DATASET  = "./data/damaged-sign3/clean_dataset"
FINAL_CLASSES  = ["traffic_sign"]


def count_from_folder(label_dir, file_list=None):
    counts = defaultdict(int)
    if file_list is None:
        file_list = [f for f in os.listdir(label_dir) if f.endswith(".txt")]
    
    for file in file_list:
        if not file.endswith(".txt"):
            continue
        with open(os.path.join(label_dir, file)) as f:
            for line in f:
                if line.strip():
                    cls = int(line.strip().split()[0])
                    counts[cls] += 1
    return counts


def get_box_sizes_px(label_dir, image_dir, file_list=None):
    sizes = []
    if file_list is None:
        file_list = [f for f in os.listdir(label_dir) if f.endswith(".txt")]
    
    for file in file_list:
        if not file.endswith(".txt"):
            continue
            
        # Find image (jpg or png)
        base = file.replace(".txt", "")
        img_path = os.path.join(image_dir, base + ".jpg")
        if not os.path.exists(img_path):
            img_path = os.path.join(image_dir, base + ".png")
            if not os.path.exists(img_path):
                continue

        with Image.open(img_path) as img:
            img_w, img_h = img.size

        with open(os.path.join(label_dir, file)) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    w_norm = float(parts[3])
                    h_norm = float(parts[4])
                    real_w = w_norm * img_w
                    real_h = h_norm * img_h
                    sizes.append(real_w * real_h)
    return sizes


# =============================================================================
# 1. Get split ratios from FINAL dataset
# =============================================================================
def count_images(split_path):
    label_dir = os.path.join(split_path, "labels")
    if not os.path.exists(label_dir):
        return 0
    return len([f for f in os.listdir(label_dir) if f.endswith(".txt")])

train_n = count_images(os.path.join(OUTPUT_DATASET, "train"))
val_n   = count_images(os.path.join(OUTPUT_DATASET, "val"))
test_n  = count_images(os.path.join(OUTPUT_DATASET, "test"))
total_n = train_n + val_n + test_n

train_ratio = train_n / total_n if total_n > 0 else 0.7
val_ratio   = val_n   / total_n if total_n > 0 else 0.15
test_ratio  = test_n  / total_n if total_n > 0 else 0.15


# =============================================================================
# 2. Logical split for clean_dataset (only has train folder)
# =============================================================================
clean_label_dir = os.path.join(CLEAN_DATASET, "train", "labels")
clean_image_dir = os.path.join(CLEAN_DATASET, "train", "images")

clean_files = [f for f in os.listdir(clean_label_dir) if f.endswith(".txt")] if os.path.exists(clean_label_dir) else []

random.seed(42)
random.shuffle(clean_files)

n = len(clean_files)
train_idx = int(n * train_ratio)
val_idx   = train_idx + int(n * val_ratio)

clean_train_files = clean_files[:train_idx]
clean_val_files   = clean_files[train_idx:val_idx]
clean_test_files  = clean_files[val_idx:]


# =============================================================================
# 3. Load data for both datasets
# =============================================================================

# --- Final Dataset ---
final_train_counts = count_from_folder(os.path.join(OUTPUT_DATASET, "train", "labels"))
final_val_counts   = count_from_folder(os.path.join(OUTPUT_DATASET, "val",   "labels"))
final_test_counts  = count_from_folder(os.path.join(OUTPUT_DATASET, "test",  "labels"))

final_train_sizes = get_box_sizes_px(
    os.path.join(OUTPUT_DATASET, "train", "labels"),
    os.path.join(OUTPUT_DATASET, "train", "images")
)
final_val_sizes = get_box_sizes_px(
    os.path.join(OUTPUT_DATASET, "val", "labels"),
    os.path.join(OUTPUT_DATASET, "val", "images")
)
final_test_sizes = get_box_sizes_px(
    os.path.join(OUTPUT_DATASET, "test", "labels"),
    os.path.join(OUTPUT_DATASET, "test", "images")
)

# --- Clean Dataset ---
clean_train_counts = count_from_folder(clean_label_dir, clean_train_files)
clean_val_counts   = count_from_folder(clean_label_dir, clean_val_files)
clean_test_counts  = count_from_folder(clean_label_dir, clean_test_files)

clean_train_sizes = get_box_sizes_px(clean_label_dir, clean_image_dir, clean_train_files)
clean_val_sizes   = get_box_sizes_px(clean_label_dir, clean_image_dir, clean_val_files)
clean_test_sizes  = get_box_sizes_px(clean_label_dir, clean_image_dir, clean_test_files)


# =============================================================================
# 4. Plot: 1 Bar Chart + 1 Histogram for clear comparison
# =============================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

x = np.arange(len(FINAL_CLASSES))
width = 0.35

# ====================== BAR CHART (Class Distribution) ======================
# Final dataset (left bars)
ax1.bar(x - width/2, 
        [final_train_counts.get(i, 0) for i in range(len(FINAL_CLASSES))], 
        width, label='Final - Train', color='skyblue', alpha=0.9)
ax1.bar(x - width/2, 
        [final_val_counts.get(i, 0)   for i in range(len(FINAL_CLASSES))], 
        width, bottom=[final_train_counts.get(i, 0) for i in range(len(FINAL_CLASSES))],
        label='Final - Val', color='orange', alpha=0.9)
ax1.bar(x - width/2, 
        [final_test_counts.get(i, 0)  for i in range(len(FINAL_CLASSES))], 
        width, 
        bottom=[final_train_counts.get(i, 0) + final_val_counts.get(i, 0) for i in range(len(FINAL_CLASSES))],
        label='Final - Test', color='green', alpha=0.9)

# Clean dataset (right bars)
ax1.bar(x + width/2, 
        [clean_train_counts.get(i, 0) for i in range(len(FINAL_CLASSES))], 
        width, label='Original - Train', color='lightblue', hatch='//')
ax1.bar(x + width/2, 
        [clean_val_counts.get(i, 0)   for i in range(len(FINAL_CLASSES))], 
        width, bottom=[clean_train_counts.get(i, 0) for i in range(len(FINAL_CLASSES))],
        label='Original - Val', color='navajowhite', hatch='//')
ax1.bar(x + width/2, 
        [clean_test_counts.get(i, 0)  for i in range(len(FINAL_CLASSES))], 
        width,
        bottom=[clean_train_counts.get(i, 0) + clean_val_counts.get(i, 0) for i in range(len(FINAL_CLASSES))],
        label='Original - Test', color='lightgreen', hatch='//')

ax1.set_xticks(x)
ax1.set_xticklabels(FINAL_CLASSES)
ax1.set_xlabel("Classes")
ax1.set_ylabel("Number of Instances")
ax1.set_title("Class Distribution Comparison\n(Final Dataset vs Original Dataset)")
ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
ax1.grid(axis='y', alpha=0.3)

# ====================== HISTOGRAM (Box Size Distribution) ======================
ax2.hist(final_train_sizes, bins=50, alpha=0.6, label='Final - Train', color='skyblue')
ax2.hist(final_val_sizes,   bins=50, alpha=0.6, label='Final - Val',   color='orange')
ax2.hist(final_test_sizes,  bins=50, alpha=0.6, label='Final - Test',  color='green')

ax2.hist(clean_train_sizes, bins=50, alpha=0.6, label='Original - Train', color='lightblue', linestyle='--', linewidth=1.2)
ax2.hist(clean_val_sizes,   bins=50, alpha=0.6, label='Original - Val',   color='navajowhite', linestyle='--', linewidth=1.2)
ax2.hist(clean_test_sizes,  bins=50, alpha=0.6, label='Original - Test',  color='lightgreen', linestyle='--', linewidth=1.2)

ax2.set_xlabel("Bounding Box Area (pixels²)")
ax2.set_ylabel("Frequency (Count)")
ax2.set_yscale('log')
ax2.set_title("Bounding Box Size Distribution Comparison")
ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.show()