import json
import os
import numpy as np
import faiss
import torch
from PIL import Image
from sentence_transformers import SentenceTransformer
from torchvision import transforms
from transformers import AutoImageProcessor, AutoModel

# --- Configuration ---
CLIP_MODEL_NAME = 'clip-ViT-B-32'
DINO_MODEL_NAME = 'facebook/dinov2-base' # 'base' is 768-dim, 'small' is 384-dim
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 1. Load CLIP
clip_model = SentenceTransformer(CLIP_MODEL_NAME, device=DEVICE)

# 2. Load DINOv2
dino_processor = AutoImageProcessor.from_pretrained(DINO_MODEL_NAME)
dino_model = AutoModel.from_pretrained(DINO_MODEL_NAME).to(DEVICE)
dino_model.eval()

# Augmentation for robustness
augmenter = transforms.Compose([
    transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0)),
])

def get_combined_embedding(image, clip_weight=0.6, dino_weight=0.4):
    # CLIP
    clip_emb = clip_model.encode(image, convert_to_numpy=True)
    clip_emb = clip_emb / np.linalg.norm(clip_emb)

    # DINOv2
    inputs = dino_processor(images=image, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = dino_model(**inputs)
        dino_emb = outputs.last_hidden_state[:, 0, :].cpu().numpy().flatten()
    dino_emb = dino_emb / np.linalg.norm(dino_emb)

    # Apply Weights (Important: use sqrt for norm preservation)
    weighted_clip = clip_emb * np.sqrt(clip_weight)
    weighted_dino = dino_emb * np.sqrt(dino_weight)

    return np.hstack((weighted_clip, weighted_dino))

# --- Processing Logic ---

with open('nsw_traffic_signs.json', 'r', encoding='utf-8') as f:
    all_signs = json.load(f)

embeddings = []
valid_signs = []

print(f"Generating combined embeddings using {DEVICE}...")

for sign in all_signs:
    img_path = sign.get('local_image_path')
    if img_path and os.path.exists(img_path):
        try:
            original_img = Image.open(img_path).convert('RGB')
            
            # 1. Encode original
            combined_vec = get_combined_embedding(original_img)
            embeddings.append(combined_vec)
            valid_signs.append(sign)
            
            # 2. Encode Augmented (Optional: reduced count to save space/time)
            for _ in range(1): 
                aug_img = augmenter(original_img)
                embeddings.append(get_combined_embedding(aug_img))
                valid_signs.append(sign)
        except Exception as e:
            print(f"Error processing {img_path}: {e}")

# Convert to FAISS-ready array
embeddings_array = np.array(embeddings).astype('float32')

# Build the FAISS Index
dimension = embeddings_array.shape[1]
print(f"Total Vector Dimension: {dimension}")

# IndexFlatIP is used for Inner Product (Cosine Similarity since we normalized)
index = faiss.IndexFlatIP(dimension)
index.add(embeddings_array)

print(f"FAISS index built with {index.ntotal} vectors.")

# Save
faiss.write_index(index, "traffic_signs_4_hybrid.index")
with open("traffic_signs_metadata_4_hybrid.json", "w", encoding="utf-8") as f:
    json.dump(valid_signs, f, ensure_ascii=False, indent=2)

print("Hybrid Index and metadata saved successfully.")