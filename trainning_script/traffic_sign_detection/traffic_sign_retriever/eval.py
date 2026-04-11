import faiss
import json
from PIL import Image, ImageOps, ImageFilter, ImageStat, ImageOps
from sentence_transformers import SentenceTransformer
from pathlib import Path
import os
import matplotlib.pyplot as plt
import requests
from io import BytesIO
import time

import torch
import torch.nn as nn
from torchvision import models, transforms
import numpy as np
from transformers import AutoImageProcessor, AutoModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Loading CLIP model and index...")
index_clip = faiss.read_index("./traffic_signs_3.index")
model_clip = SentenceTransformer('clip-ViT-B-32')
with open("./traffic_signs_metadata_5.json", "r", encoding="utf-8") as f:
    valid_signs_clip = json.load(f)

index_dict = {
    "flat": faiss.read_index("./flat.index"),
    "hnsw": faiss.read_index("./hnsw.index"),
    "ivf": faiss.read_index("./ivf.index"),
    "ivfpq": faiss.read_index("./ivfpq.index"),
}
index_dict["ivf"].nprobe = 10
index_dict["ivfpq"].nprobe = 10
index_dict["hnsw"].hnsw.efSearch = 50

print("Loading ResNet-50 model and index...")
resnet50 = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
resnet_model = nn.Sequential(*list(resnet50.children())[:-1])
resnet_model.to(device).eval()
index_resnet = faiss.read_index("./traffic_signs_4_resnet.index")
with open("./traffic_signs_metadata_4_resnet.json", "r", encoding="utf-8") as f:
    valid_signs_resnet = json.load(f)

print("Loading DINOv2 for Hybrid Retrieval...")
dino_processor = AutoImageProcessor.from_pretrained('facebook/dinov2-base')
dino_model = AutoModel.from_pretrained('facebook/dinov2-base').to(device).eval()
index_hybrid = faiss.read_index("traffic_signs_4_hybrid.index")
with open("traffic_signs_metadata_4_hybrid.json", "r", encoding="utf-8") as f:
    valid_signs_hybrid = json.load(f)

print("Loading DINOv2 Index and Metadata...")
# Make sure these filenames match what you saved in the 'Data Creation' step
index_dino = faiss.read_index("traffic_signs_dino_only.index")
with open("traffic_signs_metadata_dino_only.json", "r", encoding="utf-8") as f:
    valid_signs_dino = json.load(f)

def is_valid_crop(img, noise_threshold=85.0, bw_threshold=5.0):
    """
    Returns False if the image is too noisy or lacks color (B&W).
    """
    # 1. Check for Black & White (Saturation)
    # Convert to HSV and get the average Saturation (Index 1)
    hsv_img = img.convert('HSV')
    stat_hsv = ImageStat.Stat(hsv_img)
    avg_saturation = stat_hsv.mean[1]
    
    if avg_saturation < bw_threshold:
        # print("Skipping: Image is Black & White")
        return False

    # 2. Check for High-Frequency Noise (Salt & Pepper)
    # We use the Standard Deviation of the Grayscale version.
    # Extremely noisy images have unnaturally high variance.
    gray_img = img.convert('L')
    stat_gray = ImageStat.Stat(gray_img)
    std_dev = stat_gray.stddev[0]
    
    if std_dev > noise_threshold:
        # print(f"Skipping: Too much noise (StdDev: {std_dev:.2f})")
        return False

    # 3. Check for "Empty" or "Flat" images (all black/white)
    if std_dev < 2.0:
        return False

    return True

from PIL import Image, ImageFilter

def preprocess_for_live_feed(img, apply_sharpen=True, apply_padding=True, desired_size=224):
    """
    Flexible preprocessing for traffic sign crops.
    
    Args:
        img (PIL.Image): The input crop.
        apply_sharpen (bool): Whether to apply BICUBIC upscaling and sharpening filters.
        apply_padding (bool): Whether to pad the image to a square with a neutral background.
        desired_size (int): Target dimension for the output image.
    """
    # Basic conversion - always necessary
    img = img.convert('RGB')
    
    # 1. Scaling and Sharpening (The "Enhanced" path)
    if apply_sharpen:
        w, h = img.size
        if w < 100 or h < 100:
            # Upscale first to preserve detail before filtering
            img = img.resize((128, 128), Image.Resampling.BICUBIC)
        
        # Apply filters to make internal icons/text pop
        img = img.filter(ImageFilter.SHARPEN)
        img = img.filter(ImageFilter.DETAIL)
    
    # 2. Resizing and Padding logic
    if apply_padding:
        # Standardize size while maintaining aspect ratio
        img.thumbnail((desired_size, desired_size), Image.Resampling.LANCZOS)
        
        # Create neutral gray background
        new_img = Image.new("RGB", (desired_size, desired_size), (128, 128, 128))
        upper_left = ((desired_size - img.size[0]) // 2, (desired_size - img.size[1]) // 2)
        new_img.paste(img, upper_left)
        return new_img
    else:
        # Basic path: Just resize directly to the target size without padding
        # Note: This will stretch the image if the original isn't square
        return img.resize((desired_size, desired_size), Image.Resampling.LANCZOS)

def retrieve_top_k_clip(query_img: Image.Image,k: int = 3, index_type="flat"):
    """CLIP retrieval – uses existing model/index/metadata."""
    if index_type not in index_dict:
        raise ValueError(f"Unknown index_type: {index_type}")
    
    index = index_dict[index_type]

    processed_img = preprocess_for_live_feed(query_img)
    
    query_vector = model_clip.encode(processed_img).astype('float32').reshape(1, -1)
    faiss.normalize_L2(query_vector)
    
    distances, indices = index.search(query_vector, k)
    
    matches = []
    for i in range(k):
        match_idx = indices[0][i]
        if match_idx < 0:
            continue
        
        sign_data = valid_signs_clip[match_idx]
        
        matches.append({
            "rank": i + 1,
            "similarity": float(distances[0][i]),
            "category": sign_data.get("category"),
            "sign_no": sign_data.get("Sign No"),
            "original_url": sign_data.get("original_url"),
            "description": sign_data.get("Descriptions")
        })
    
    return matches

def get_resnet_embedding(processed_img: Image.Image):
    """ResNet-50 feature extraction using the SAME visual preprocessing as CLIP + ImageNet normalization."""
    resnet_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    img_t = resnet_transform(processed_img).unsqueeze(0).to(device)
    with torch.no_grad():
        embedding = resnet_model(img_t)
    return embedding.cpu().numpy().flatten().astype('float32')


def retrieve_top_k_resnet(query_img: Image.Image, k: int = 3):
    """ResNet-50 retrieval – uses same live-feed preprocessing for fair comparison."""
    processed_img = preprocess_for_live_feed(query_img, apply_sharpen=False, apply_padding=False)
    
    query_vector = get_resnet_embedding(processed_img).reshape(1, -1)
    faiss.normalize_L2(query_vector)
    
    distances, indices = index_resnet.search(query_vector, k)
    
    matches = []
    for i in range(k):
        match_idx = indices[0][i]
        if match_idx < 0:
            continue
        sign_data = valid_signs_resnet[match_idx]
        
        matches.append({
            "rank": i + 1,
            "similarity": float(distances[0][i]),
            "category": sign_data.get("category"),
            "sign_no": sign_data.get("Sign No"),
            "original_url": sign_data.get("original_url"),
            "description": sign_data.get("Descriptions")
        })
    return matches

def get_hybrid_embedding(processed_img: Image.Image, clip_weight=0.6, dino_weight=0.4):
    """Concatenates weighted CLIP and DINOv2 features."""
    
    # 1. CLIP part
    
    clip_emb = model_clip.encode(processed_img, convert_to_numpy=True)
    clip_emb = clip_emb / np.linalg.norm(clip_emb) # Ensure unit length
    
    # 2. DINOv2 part
    inputs = dino_processor(images=processed_img, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = dino_model(**inputs)
        dino_emb = outputs.last_hidden_state[:, 0, :].cpu().numpy().flatten()
    dino_emb = dino_emb / np.linalg.norm(dino_emb) # Ensure unit length

    # 3. Apply Weights
    # We use sqrt so the resulting vector stays normalized for FAISS IndexFlatIP
    weighted_clip = clip_emb * np.sqrt(clip_weight)
    weighted_dino = dino_emb * np.sqrt(dino_weight)

    final_emb = np.hstack((weighted_clip, weighted_dino))
    return final_emb.astype('float32')

# --- 3. RETRIEVAL FUNCTIONS ---

def retrieve_top_k_hybrid(query_img: Image.Image, k: int = 3):
    processed_img = preprocess_for_live_feed(query_img, apply_sharpen=True, apply_padding=True)
    query_vector = get_hybrid_embedding(processed_img).reshape(1, -1)
    
    # We don't normalize the final concatenated vector here because 
    # IndexFlatIP handles inner product and components were pre-normalized.
    distances, indices = index_hybrid.search(query_vector, k)
    
    matches = []
    for i in range(k):
        match_idx = indices[0][i]
        if match_idx < 0: continue
        sign_data = valid_signs_hybrid[match_idx]
        matches.append({
            "rank": i + 1,
            "similarity": float(distances[0][i]),
            "category": sign_data.get("category"),
            "description": sign_data.get("Descriptions"),
            "original_url": sign_data.get("original_url")
        })
    return matches

def get_dino_only_embedding(processed_img: Image.Image):
    """
    Extracts a unit-normalized DINOv2 embedding.
    Ensures the vector is float32 and reshaped for FAISS.
    """
    inputs = dino_processor(images=processed_img, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = dino_model(**inputs)
        # Using the [CLS] token (index 0)
        dino_emb = outputs.last_hidden_state[:, 0, :].cpu().numpy().flatten()
    
    # Normalize for Cosine Similarity (IndexFlatIP)
    norm = np.linalg.norm(dino_emb)
    if norm > 0:
        dino_emb = dino_emb / norm
        
    return dino_emb.astype('float32')

def retrieve_top_k_dino(query_img: Image.Image, k: int = 3):
    """
    DINOv2-only retrieval function.
    Make sure apply_sharpen and apply_padding match your INDEXING settings.
    """
    # Preprocess the crop
    processed_img = preprocess_for_live_feed(query_img, apply_sharpen=True, apply_padding=True)
    
    # Generate Query Vector
    query_vector = get_dino_only_embedding(processed_img).reshape(1, -1)
    
    # Search the DINO-specific index
    # Note: Ensure you have loaded 'index_dino' and 'valid_signs_dino'
    distances, indices = index_dino.search(query_vector, k)
    
    matches = []
    for i in range(k):
        match_idx = indices[0][i]
        if match_idx < 0: 
            continue
            
        sign_data = valid_signs_dino[match_idx]
        matches.append({
            "rank": i + 1,
            "similarity": float(distances[0][i]),
            "category": sign_data.get("category"),
            "description": sign_data.get("Descriptions") or sign_data.get("description"),
            "original_url": sign_data.get("original_url"),
            "sign_no": sign_data.get("Sign No")
        })
    return matches

def save_error_plot(query_crop, gt_name, match_data, save_path):
    url = match_data.get('original_url')
    if not url:
        print("Skip: No URL provided for this match.")
        return

    # 1. Add headers to avoid being blocked by the server
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    }

    try:
        response = requests.get(url, headers=headers, timeout=10)
        
        # 2. Check if the response is actually an image
        content_type = response.headers.get('Content-Type', '')
        if 'image' not in content_type:
            print(f"⚠️ URL did not return an image. Type: {content_type} | URL: {url}")
            return

        match_img = Image.open(BytesIO(response.content)).convert("RGB")
        
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        
        # Plot Query
        axes[0].imshow(query_crop)
        axes[0].set_title(f"QUERY (GT): {gt_name}", color='red', fontsize=9)
        axes[0].axis('off')
        
        # Plot Result
        axes[1].imshow(match_img)
        axes[1].set_title(f"MATCH: {match_data['description']}\nScore: {match_data['similarity']:.3f}", 
                         color='blue', fontsize=9)
        axes[1].axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        
    except Exception as e:
        print(f"❌ Error plotting from {url}: {e}")

def run_benchmark(
    dataset_dir: str,
    class_names: list[str],
    retrieve_func,
    model_name: str = "Model",
    min_crop_dim: int = 32,
    max_images: int = 100,
    top_k: int = 3
):
    data_root = Path(dataset_dir)
    splits = ["train", "valid", "test"]

    image_paths = []
    for split in splits:
        split_img_dir = data_root / split / "images"
        if split_img_dir.exists():
            image_paths.extend(list(split_img_dir.glob("*.jpg")))
            image_paths.extend(list(split_img_dir.glob("*.jpeg")))
            image_paths.extend(list(split_img_dir.glob("*.png")))
    
    image_paths = sorted(image_paths)
    if max_images is not None:
        image_paths = image_paths[:max_images]
    
    # --- Metrics Initialization ---
    total_queries = 0
    top1_correct = 0
    topk_correct = 0
    mrr_sum = 0.0
    latencies = []
    skipped_small = 0
    processed_queries = 0
    
    error_folder = f"error_analysis_{model_name.lower()}"
    os.makedirs(error_folder, exist_ok=True)
    error_count = 0
    max_errors_to_plot = 200

    print(f"Starting benchmark on {len(image_paths)} images for {model_name}...")

    for img_path in image_paths:
        label_dir = img_path.parent.parent / "labels"
        label_path = label_dir / f"{img_path.stem}.txt"
        if not label_path.exists():
            continue
            
        try:
            img = Image.open(img_path).convert("RGB")
            W, H = img.size
            with open(label_path, "r", encoding="utf-8") as f:
                label_lines = f.readlines()
            
            for line in label_lines:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                cls_id = int(parts[0])
                if cls_id >= len(class_names):
                    continue
                    
                gt_name = class_names[cls_id]
                gt_name_clean = gt_name.lower().strip()
                
                x_c, y_c, w_norm, h_norm = map(float, parts[1:5])
                x1 = max(0, int((x_c - w_norm / 2) * W))
                y1 = max(0, int((y_c - h_norm / 2) * H))
                x2 = min(W, int((x_c + w_norm / 2) * W))
                y2 = min(H, int((y_c + h_norm / 2) * H))
                
                crop = img.crop((x1, y1, x2, y2))
                
                if crop.width < min_crop_dim or crop.height < min_crop_dim or not is_valid_crop(crop):
                    skipped_small += 1
                    continue

                # --- Retrieval with Latency Timing ---
                start_time = time.perf_counter()
                matches = retrieve_func(crop, k=top_k)
                end_time = time.perf_counter()
                
                if not matches:
                    continue
                
                latencies.append(end_time - start_time)
                total_queries += 1
                
                # --- Rank & MRR Calculation ---
                found_rank = 0
                for idx, m in enumerate(matches):
                    match_desc = m["description"].lower().strip()
                    if gt_name_clean in match_desc or match_desc in gt_name_clean:
                        found_rank = idx + 1
                        break
                
                if found_rank == 1:
                    top1_correct += 1
                
                if found_rank > 0:
                    topk_correct += 1
                    mrr_sum += (1.0 / found_rank)
                
                # Error plotting for Top-1 failures
                if found_rank != 1 and error_count < max_errors_to_plot:
                    error_count += 1
                    save_path = f"{error_folder}/error_{error_count}_{img_path.stem}.png"
                    save_error_plot(crop, gt_name, matches[0], save_path)
                
                processed_queries += 1
                if processed_queries % 20 == 0:
                    print(f"  → Processed {processed_queries} valid crops for {model_name}...")
                    
        except Exception as e:
            print(f"⚠️ Error processing {img_path.name}: {e}")

    # --- Final Report ---
    if total_queries == 0:
        print(f"\n❌ No valid crops were processed for {model_name}.")
        return {"model": model_name, "valid_queries": 0, "top1_acc": 0, "topk_acc": 0, "mrr": 0, "avg_latency_ms": 0}

    avg_latency = (sum(latencies) / len(latencies)) * 1000
    mrr = mrr_sum / total_queries

    print("\n" + "="*70)
    print(f"{model_name} BENCHMARK RESULTS")
    print("="*70)
    print(f"Valid query crops         : {total_queries}")
    print(f"Skipped (small/noisy)     : {skipped_small}")
    print("-" * 40)
    print(f"Top-1 Accuracy            : {top1_correct / total_queries * 100:.2f}%")
    print(f"Top-{top_k} Accuracy          : {topk_correct / total_queries * 100:.2f}%")
    print(f"Mean Reciprocal Rank (MRR): {mrr:.4f}")
    print(f"Avg Latency per Query     : {avg_latency:.2f} ms")
    print("="*70)

    return {
        "model": model_name,
        "valid_queries": total_queries,
        "top1_acc": top1_correct / total_queries * 100,
        "topk_acc": topk_correct / total_queries * 100,
        "mrr": mrr,
        "avg_latency_ms": avg_latency,
    }

# ====================== CONFIG & RUN ======================
if __name__ == "__main__":
    TEST_IMAGES_DIR = "./benchmark_dataset/"
    
    # Your exact class list from data.yaml (order must match class IDs in labels)
    CLASS_NAMES = [
        '10 km-h Speed Limit', '100 km-h Speed Limit', '20 km-h Speed Limit',
        '30 km-h Speed Limit', '40 km-h Speed Limit', '50 km-h Speed Limit',
        '60 km-h Speed Limit', '80 km-h Speed Limit', 'Added Lane -left-',
        'Advisory speed 20', 'Bicycle Lane', 'Bicycles Only', 'Bus Lane',
        'Curve to left', 'Curve to right', 'End Clearway', 'End Roadwork',
        'End Shared Zone', 'End of road Curve marker', 'Give Way to Pedestrians',
        'Give way', 'Give way at roundabout', 'Island curve marker -right-',
        'Keep Left', 'Keep Right', 'Merging Traffic -left-', 'No Bicycles',
        'No Entry', 'No Left Turn', 'No Pedestrians', 'No Right Turn',
        'No Trucks', 'No U-turn', 'One Way -left-', 'One Way -right-',
        'Pass either side', 'Pedestrian Crossing', 'Pedestrian Crossing Ahead',
        'Pedestrian Crossing Ahead on Side Road -veer left-', 'Pedestrians',
        'Prepare to Stop', 'Hook Turn Only', 'Road Hump',
        'Road Narrows', 'Road Safety Cameras Operate In This Area',
        'Roundabout Ahead', 'Safety Zone', 'Shared Path', 'Shared Zone',
        'Slow', 'State route shield', 'Stop',
        'Stop', 'Traffic Lights ahead',
        'Tram Only', 'Tram Speed', 'Trucks Crossing or Entering',
        'Turn Left', 'Turn Right', 'Workers Ahead', 'steep descent', 'train speed 10'
    ]
    
    # Hyper-parameters (feel free to tweak)
    MIN_CROP_DIM = 48          # pixels — increase if you want stricter filtering
    MAX_IMAGES_TO_TEST = None   # set to None to run the entire test set
    TOP_K = 3

    clip_metrics_flat = run_benchmark(
    dataset_dir=TEST_IMAGES_DIR,
    class_names=CLASS_NAMES,
    retrieve_func=lambda img, k: retrieve_top_k_clip(img, k, index_type="flat"),
    model_name="CLIP_Flat",
    min_crop_dim=MIN_CROP_DIM,
    max_images=MAX_IMAGES_TO_TEST,
    top_k=TOP_K
    )

    clip_metrics_hnsw = run_benchmark(
    dataset_dir=TEST_IMAGES_DIR,
    class_names=CLASS_NAMES,
    retrieve_func=lambda img, k: retrieve_top_k_clip(img, k, index_type="hnsw"),
    model_name="CLIP_HNSW",
    min_crop_dim=MIN_CROP_DIM,
    max_images=MAX_IMAGES_TO_TEST,
    top_k=TOP_K
    )

    clip_metrics_ivf = run_benchmark(
    dataset_dir=TEST_IMAGES_DIR,
    class_names=CLASS_NAMES,
    retrieve_func=lambda img, k: retrieve_top_k_clip(img, k, index_type="ivf"),
    model_name="CLIP_IVF",
    min_crop_dim=MIN_CROP_DIM,
    max_images=MAX_IMAGES_TO_TEST,
    top_k=TOP_K
    )

    clip_metrics_ivfpq = run_benchmark(
    dataset_dir=TEST_IMAGES_DIR,
    class_names=CLASS_NAMES,
    retrieve_func=lambda img, k: retrieve_top_k_clip(img, k, index_type="ivfpq"),
    model_name="CLIP_IVFPQ",
    min_crop_dim=MIN_CROP_DIM,
    max_images=MAX_IMAGES_TO_TEST,
    top_k=TOP_K
    )

    results = [clip_metrics_flat, clip_metrics_hnsw, clip_metrics_ivf, clip_metrics_ivfpq]
    
    print("\n" + "="*110)
    print(f"{'Metric':<25} {'FLAT':<15} {'HNSW':<15} {'IVF':<15} {'IVFPQ':<15} {'Winner':<15}")
    print("-" * 110)

    comparison_keys = [
        ("top1_acc", "Top-1 Accuracy (%)", True),
        ("topk_acc", f"Top-{TOP_K} Accuracy (%)", True),
        ("mrr", "MRR", True),
        ("avg_latency_ms", "Latency (ms)", False)
    ]

    for key, label, higher_better in comparison_keys:
        v_clip = clip_metrics_flat[key]
        v_hnsw = clip_metrics_hnsw[key]
        v_ivf = clip_metrics_ivf[key]
        v_ivfpq = clip_metrics_ivfpq[key]

        vals = [v_clip, v_hnsw, v_ivf, v_ivfpq]
        if higher_better:
            win_idx = np.argmax(vals)
        else:
            win_idx = np.argmin(vals)
        
        names = ["FLAT", "HNSW", "IVF", "IVFPQ"]
        winner = names[win_idx]

        print(f"{label:<25} {v_clip:<15.2f} {v_hnsw:<15.2f} {v_ivf:<15.2f} {v_ivfpq:<15.2f} {winner:<15}")

    # dino_metrics = run_benchmark(
    #     TEST_IMAGES_DIR, 
    #     CLASS_NAMES, 
    #     retrieve_top_k_dino,
    #     "DINOv2", 
    #     MIN_CROP_DIM,
    #     MAX_IMAGES_TO_TEST,
    #     TOP_K)

    # hybrid_metrics = run_benchmark(
    #     TEST_IMAGES_DIR, 
    #     CLASS_NAMES, 
    #     retrieve_top_k_hybrid,
    #     "Hybrid-CLIP-DINO", 
    #     MIN_CROP_DIM,
    #     MAX_IMAGES_TO_TEST,
    #     TOP_K)

    # clip_metrics = run_benchmark(
    #     dataset_dir=TEST_IMAGES_DIR,
    #     class_names=CLASS_NAMES,
    #     retrieve_func=retrieve_top_k_clip,
    #     model_name="CLIP",
    #     min_crop_dim=MIN_CROP_DIM,
    #     max_images=MAX_IMAGES_TO_TEST,
    #     top_k=TOP_K
    # )

    # # Run ResNet-50 benchmark
    # resnet_metrics = run_benchmark(
    #     dataset_dir=TEST_IMAGES_DIR,
    #     class_names=CLASS_NAMES,
    #     retrieve_func=retrieve_top_k_resnet,
    #     model_name="ResNet-50",
    #     min_crop_dim=MIN_CROP_DIM,
    #     max_images=MAX_IMAGES_TO_TEST,
    #     top_k=TOP_K
    # )
    # # ====================== SIDE-BY-SIDE COMPARISON ======================
    # results = [clip_metrics, dino_metrics, resnet_metrics, hybrid_metrics]
    
    # print("\n" + "="*110)
    # print(f"{'Metric':<25} {'CLIP':<15} {'DINOv2':<15} {'ResNet-50':<15} {'Hybrid (C+D)':<15} {'Winner':<15}")
    # print("-" * 110)

    # comparison_keys = [
    #     ("top1_acc", "Top-1 Accuracy (%)", True),
    #     ("topk_acc", f"Top-{TOP_K} Accuracy (%)", True),
    #     ("mrr", "MRR", True),
    #     ("avg_latency_ms", "Latency (ms)", False)
    # ]

    # for key, label, higher_better in comparison_keys:
    #     v_clip = clip_metrics[key]
    #     v_res = resnet_metrics[key]
    #     v_hyb = hybrid_metrics[key]
    #     v_dino = dino_metrics[key]
        
    #     vals = [v_clip, v_dino, v_res, v_hyb]
    #     if higher_better:
    #         win_idx = np.argmax(vals)
    #     else:
    #         win_idx = np.argmin(vals)
        
    #     names = ["CLIP", "DINOv2", "ResNet-50", "Hybrid"]
    #     winner = names[win_idx]

    #     print(f"{label:<25} {v_clip:<15.2f} {v_dino:<15.2f} {v_res:<15.2f} {v_hyb:<15.2f} {winner:<15}")