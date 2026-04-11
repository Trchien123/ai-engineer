import faiss
import json
from PIL import Image, ImageOps, ImageFilter
from sentence_transformers import SentenceTransformer

index = faiss.read_index("traffic_signs_3.index")
model = SentenceTransformer('clip-ViT-B-32')


# Load metadata
with open("traffic_signs_metadata_3.json", "r", encoding="utf-8") as f:
    valid_signs = json.load(f)

def preprocess_for_live_feed(img):
    """Enhanced preprocessing specifically for tiny crops (e.g., 64x58)."""
    img = img.convert('RGB')
    
    # 1. If the image is tiny, use a sharper resampling method
    w, h = img.size
    if w < 100 or h < 100:
        # Upscale first before filtering to preserve what little detail exists
        img = img.resize((128, 128), Image.Resampling.BICUBIC)
    
    # 2. Adaptive Sharpening
    # Over-sharpening tiny images helps CLIP see the internal icons/numbers
    img = img.filter(ImageFilter.SHARPEN)
    img = img.filter(ImageFilter.DETAIL) # Extra pass for internal sign details
    
    # 3. Square Padding with 'Edge' strategy
    # Instead of black (0,0,0), let's use the average color of the border 
    # to make the sign look more natural to the CLIP model.
    desired_size = 224
    img.thumbnail((desired_size, desired_size), Image.Resampling.LANCZOS)
    
    # Create a new background with a neutral gray or blurred version 
    # (CLIP likes context, not just black voids)
    new_img = Image.new("RGB", (desired_size, desired_size), (128, 128, 128))
    upper_left = ((desired_size - img.size[0]) // 2, (desired_size - img.size[1]) // 2)
    new_img.paste(img, upper_left)
    
    return new_img 

def search_top_k(query_image_path, k=3):
    # Load and Preprocess
    query_img = Image.open(query_image_path)
    processed_img = preprocess_for_live_feed(query_img)
    
    # Encode the image
    query_vector = model.encode(processed_img).astype('float32').reshape(1, -1)
    
    # CRITICAL: Normalize for IndexFlatIP (Cosine Similarity)
    faiss.normalize_L2(query_vector)
    
    # Search the index
    # Distances will now be similarity scores (closer to 1.0 is better)
    distances, indices = index.search(query_vector, k)
    
    print(f"\nTop {k} Matches for input image:")
    for i in range(k):
        match_idx = indices[0][i]
        
        # Guard against index errors
        if match_idx < 0: continue 
            
        sign_data = valid_signs[match_idx]
        
        # Note: With IndexFlatIP, higher distance = higher similarity
        print(f"--- Match {i+1} (Similarity Score: {distances[0][i]:.4f}) ---")
        print(f"Category: {sign_data.get('category')}")
        print(f"Sign No: {sign_data.get('Sign No')}")
        print(f"Description: {sign_data.get('Descriptions')}")

search_top_k("./test (1).jpg", k=3)