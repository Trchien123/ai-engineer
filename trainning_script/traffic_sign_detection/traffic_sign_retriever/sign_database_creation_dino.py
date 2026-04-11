import json
import os
import numpy as np
import faiss
import torch
from PIL import Image, ImageFilter
from torchvision import transforms
from transformers import AutoImageProcessor, AutoModel

# --- Configuration ---
DINO_MODEL_NAME = 'facebook/dinov2-base' 
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 1. Load DINOv2
print(f"Loading DINOv2 on {DEVICE}...")
dino_processor = AutoImageProcessor.from_pretrained(DINO_MODEL_NAME)
dino_model = AutoModel.from_pretrained(DINO_MODEL_NAME).to(DEVICE)
dino_model.eval()

# 2. Augmentation for robustness
augmenter = transforms.Compose([
    transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0)),
])

def preprocess_image(img, apply_sharpen=True, apply_padding=True, desired_size=224):
    """Identical preprocessing to the retrieval function."""
    img = img.convert('RGB')
    if apply_sharpen:
        w, h = img.size
        if w < 100 or h < 100:
            img = img.resize((128, 128), Image.Resampling.BICUBIC)
        img = img.filter(ImageFilter.SHARPEN)
        img = img.filter(ImageFilter.DETAIL)
    
    if apply_padding:
        img.thumbnail((desired_size, desired_size), Image.Resampling.LANCZOS)
        new_img = Image.new("RGB", (desired_size, desired_size), (128, 128, 128))
        upper_left = ((desired_size - img.size[0]) // 2, (desired_size - img.size[1]) // 2)
        new_img.paste(img, upper_left)
        return new_img
    else:
        return img.resize((desired_size, desired_size), Image.Resampling.LANCZOS)

def get_dino_embedding(image):
    """Generates a unit-normalized DINOv2 embedding."""
    # Ensure image is preprocessed before DINO processor
    processed_img = preprocess_image(image)
    
    inputs = dino_processor(images=processed_img, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = dino_model(**inputs)
        # Extract [CLS] token
        dino_emb = outputs.last_hidden_state[:, 0, :].cpu().numpy().flatten()
    
    # Normalize to unit length for Cosine Similarity (IndexFlatIP)
    norm = np.linalg.norm(dino_emb)
    return (dino_emb / norm) if norm > 0 else dino_emb

# --- Processing Logic ---

with open('nsw_traffic_signs.json', 'r', encoding='utf-8') as f:
    all_signs = json.load(f)

embeddings = []
valid_signs = []

print("Generating DINOv2 embeddings...")

for sign in all_signs:
    img_path = sign.get('local_image_path')
    if img_path and os.path.exists(img_path):
        try:
            original_img = Image.open(img_path).convert('RGB')
            
            # 1. Encode original
            emb = get_dino_embedding(original_img)
            embeddings.append(emb)
            valid_signs.append(sign)
            
            # 2. Encode Augmented
            for _ in range(1): 
                aug_img = augmenter(original_img)
                embeddings.append(get_dino_embedding(aug_img))
                valid_signs.append(sign)
                
        except Exception as e:
            print(f"Error processing {img_path}: {e}")

# Convert to FAISS-ready array
embeddings_array = np.array(embeddings).astype('float32')

# Build the FAISS Index
dimension = embeddings_array.shape[1]
print(f"Embedding Dimension: {dimension}")

# Use Inner Product (effectively Cosine Similarity since vectors are normalized)
index = faiss.IndexFlatIP(dimension)
index.add(embeddings_array)

print(f"FAISS index built with {index.ntotal} vectors.")

# Save with updated naming convention
faiss.write_index(index, "traffic_signs_dino_only.index")
with open("traffic_signs_metadata_dino_only.json", "w", encoding="utf-8") as f:
    json.dump(valid_signs, f, ensure_ascii=False, indent=2)

print("DINO-only Index and metadata saved successfully.")