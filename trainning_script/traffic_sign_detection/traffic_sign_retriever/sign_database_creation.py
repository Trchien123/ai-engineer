import json
import os
import numpy as np
import faiss
from PIL import Image
from sentence_transformers import SentenceTransformer
from torchvision import transforms

EXCLUDED_CATEGORIES = {}

augmenter = transforms.Compose([
    # transforms.RandomPerspective(distortion_scale=0.2, p=0.5), # Road-view angle
    transforms.ColorJitter(brightness=0.2, contrast=0.2), # Varying light
    transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0)), # Mimic motion blur
])

model = SentenceTransformer('clip-ViT-B-32')
with open('nsw_traffic_signs.json', 'r', encoding='utf-8') as f:
    all_signs = json.load(f)

embeddings = []
valid_signs = []

print("Generating embeddings for FAISS index...")
for sign in all_signs:
    category = sign.get('category', '')
    if category in EXCLUDED_CATEGORIES:
        continue

    img_path = sign.get('local_image_path')
    if img_path and os.path.exists(img_path):
        original_img = Image.open(img_path).convert('RGB')
        
        # 1. Encode original
        embeddings.append(model.encode(original_img))
        valid_signs.append(sign)
        
        # 2. Encode 2-3 augmented versions
        for _ in range(1): 
            aug_img = augmenter(original_img)
            embeddings.append(model.encode(aug_img))
            valid_signs.append(sign) # Map back to the same metadata

# Convert list to a specialized Numpy array for FAISS
embeddings_array = np.array(embeddings).astype('float32')
faiss.normalize_L2(embeddings_array) 
dimension = embeddings_array.shape[1]

#Flat
index_flat = faiss.IndexFlatIP(dimension)
index_flat.add(embeddings_array)
faiss.write_index(index_flat, "flat.index")

# HNSW
index_hnsw = faiss.IndexHNSWFlat(dimension, 32)  # 32 = M (graph connectivity)
index_hnsw.hnsw.efConstruction = 200
index_hnsw.add(embeddings_array)
faiss.write_index(index_hnsw, "hnsw.index")

# IVF
nlist = 100  # number of clusters
quantizer = faiss.IndexFlatIP(dimension)

index_ivf = faiss.IndexIVFFlat(quantizer, dimension, nlist, faiss.METRIC_INNER_PRODUCT)

index_ivf.train(embeddings_array)  # MUST TRAIN FIRST
index_ivf.add(embeddings_array)

faiss.write_index(index_ivf, "ivf.index")

# PQ
nlist = 64
m = 8  # number of subquantizers

quantizer = faiss.IndexFlatIP(dimension)
index_ivfpq = faiss.IndexIVFPQ(quantizer, dimension, nlist, m, 8)

index_ivfpq.train(embeddings_array)
index_ivfpq.add(embeddings_array)

faiss.write_index(index_ivfpq, "ivfpq.index")


# index = faiss.IndexFlatIP(dimension) 
# index.add(embeddings_array)

# print(f"FAISS index built with {index.ntotal} signs.")

# faiss.write_index(index, "traffic_signs_4.index")
# Save metadata (mapping vectors → signs)
with open("traffic_signs_metadata_5.json", "w", encoding="utf-8") as f:
    json.dump(valid_signs, f, ensure_ascii=False, indent=2)

print("Index and metadata saved.")

