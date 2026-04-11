import json
import os
import numpy as np
import faiss
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

# 1. Setup the ResNet-50 Model for Feature Extraction
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Load weights (IMAGENET1K_V2 is the modern standard)
resnet50 = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
# Remove the last fully connected layer to get features instead of class IDs
model = nn.Sequential(*list(resnet50.children())[:-1])
model.to(device)
model.eval()

# 2. ResNet Preprocessing (Critical: Standard CNNs need specific normalization)
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Your original augmenter
augmenter = transforms.Compose([
    transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0)),
])

def get_embedding(img):
    """Helper to get a vector from ResNet-50"""
    img_t = preprocess(img).unsqueeze(0).to(device)
    with torch.no_grad():
        embedding = model(img_t)
    return embedding.cpu().numpy().flatten()

# --- Main Processing Loop ---

with open('nsw_traffic_signs.json', 'r', encoding='utf-8') as f:
    all_signs = json.load(f)

embeddings = []
valid_signs = []
EXCLUDED_CATEGORIES = {}

print("Generating ResNet-50 embeddings for FAISS...")
for sign in all_signs:
    if sign.get('category') in EXCLUDED_CATEGORIES:
        continue

    img_path = sign.get('local_image_path')
    if img_path and os.path.exists(img_path):
        original_img = Image.open(img_path).convert('RGB')
        
        # 1. Encode original
        embeddings.append(get_embedding(original_img))
        valid_signs.append(sign)
        
        # 2. Encode augmented version
        aug_img = augmenter(original_img)
        embeddings.append(get_embedding(aug_img))
        valid_signs.append(sign)

# --- FAISS Indexing ---

embeddings_array = np.array(embeddings).astype('float32')
faiss.normalize_L2(embeddings_array) 

# ResNet-50 features are 2048-dimensional
dimension = embeddings_array.shape[1] 
index = faiss.IndexFlatIP(dimension) 
index.add(embeddings_array)

print(f"FAISS index built with {index.ntotal} signs. Dimension: {dimension}")

faiss.write_index(index, "traffic_signs_4_resnet.index")
with open("traffic_signs_metadata_4_resnet.json", "w", encoding="utf-8") as f:
    json.dump(valid_signs, f, ensure_ascii=False, indent=2)

print("ResNet-50 Index and metadata saved.")