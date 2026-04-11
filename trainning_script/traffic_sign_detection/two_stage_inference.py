import cv2
from ultralytics import YOLO
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import efficientnet_v2_s
from PIL import Image
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import json
from PIL import Image, ImageOps, ImageFilter


YOLO_MODEL_PATH = "./yolov26_1_class.pt"
CLF_MODEL_PATH = "./classification_damage_multi_class/EffnetV2_multilabel.pth"
FAISS_INDEX_PATH = "./data_crawling/traffic_signs_3.index"
METADATA_PATH = "./data_crawling/traffic_signs_metadata_3.json"

MODE = "IMAGE"
INPUT_PATH = "./test_image(4).jpg"  # Or "./test_video_trim.mp4"
OUTPUT_PATH = "output_result.jpg" if MODE == "IMAGE" else "output_with_damage.mp4"

CLASS_NAMES = [
    "bent",
    "broken_sheet",
    "crack",
    "graffiti",
    "normal",
    "paint_loss",
    "rust",
    "scratch"
]

RETRIEVER_THRESHOLD = 0.75

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

def load_retriever():
    index = faiss.read_index(FAISS_INDEX_PATH)
    clip_model = SentenceTransformer('clip-ViT-B-32')
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    return index, clip_model, metadata

def get_sign_metadata(crop_pil, index, clip_model, metadata):
    """Retrieves sign info or returns 'Unknown' if distance is too high."""
    processed_img = preprocess_for_live_feed(crop_pil)
    query_vector = clip_model.encode(processed_img).astype('float32').reshape(1, -1)
    faiss.normalize_L2(query_vector)
    distances, indices = index.search(query_vector, 1) # Get top 1
    
    dist = distances[0][0]
    idx = indices[0][0]
    
    if dist < RETRIEVER_THRESHOLD:
        return {"Sign No": "Unknown", "Descriptions": "Unknown Sign Type"}, dist

    return metadata[idx], dist

def load_effnetv2_multilabel(model_path, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    # Rebuild the architecture
    model = efficientnet_v2_s(weights=None)

    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(in_features, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(inplace=True),

        nn.Dropout(p=0.3),
        nn.Linear(512, 128),
        nn.BatchNorm1d(128),
        nn.ReLU(inplace=True),

        nn.Linear(128, len(CLASS_NAMES))
    )

    # Load weights
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)

    model = model.to(device)
    model.eval()

    return model

def predict_crop(crop_pil: Image.Image, model, device, threshold=0.5):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    tensor = transform(crop_pil).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.sigmoid(outputs).squeeze(0)  # 🔥 sigmoid for multilabel

    preds = (probs > threshold).int()

    # Get all predicted labels
    predicted_labels = [
        (CLASS_NAMES[i], probs[i].item())
        for i in range(len(CLASS_NAMES)) if preds[i] == 1
    ]

    return predicted_labels, probs.cpu()

def process_frame(frame, yolo_model, clf_model, retriever_assets, device):
    index, clip_model, metadata = retriever_assets
    results = yolo_model.predict(frame, conf=0.2)
    # results = yolo_model.track(frame, persist=True)
    annotated_frame = frame.copy()

    if results[0].boxes is None: return annotated_frame

    for box in results[0].boxes:
        coords = box.xyxy[0].cpu().numpy().astype(int)
        x1, y1, x2, y2 = coords
        yolo_conf = float(box.conf[0])
        if yolo_conf < 0.5: continue

        crop_bgr = frame[y1:y2, x1:x2]
        if crop_bgr.size == 0: continue
        crop_pil = Image.fromarray(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB))

        # 1. Classify Damage (EffNet)
        damage_preds, probs = predict_crop(crop_pil, clf_model, device)
        if len(damage_preds) == 0:
            damage_text = "No Damage"
        else:
            damage_text = ", ".join([f"{name}:{conf:.2f}" for name, conf in damage_preds])

        # 2. Identify Sign Type (Retriever)
        sign_info, dist = get_sign_metadata(crop_pil, index, clip_model, metadata)
        sign_id = sign_info.get('Descriptions', 'N/A')

        # 3. UI Drawing
        color = (0, 255, 0) if len(damage_preds) == 0 else (0, 0, 255)
        # color = (0, 255, 0) if damage_class == "normal" else (0, 0, 255)
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
        
        # Multi-line label
        # label_top = f"ID: {sign_id} | {damage_class}"
        # label_bot = f"Dist: {dist:.2f} | Conf: {damage_conf:.2f}"
        label_top = f"ID: {sign_id}"
        label_bot = f"{damage_text}"
        
        cv2.putText(annotated_frame, label_top, (x1, y1 - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        cv2.putText(annotated_frame, label_bot, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
    return annotated_frame

if __name__ == "__main__":
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Loading Models on {DEVICE}...")
    model_yolo = YOLO(YOLO_MODEL_PATH)
    clf_model = load_effnetv2_multilabel(CLF_MODEL_PATH, DEVICE)
    retriever_assets = load_retriever()

    if MODE == "IMAGE":
        frame = cv2.imread(INPUT_PATH)
        if frame is None:
            print(f"Error: Could not read image at {INPUT_PATH}")
        else:
            result_img = process_frame(frame, model_yolo, clf_model, retriever_assets, DEVICE)
            cv2.imwrite(OUTPUT_PATH, result_img)
            cv2.imshow("Result", result_img)
            print(f"Saved image to {OUTPUT_PATH}")
            cv2.waitKey(0)

    elif MODE == "VIDEO":
        cap = cv2.VideoCapture(INPUT_PATH)
        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps    = cap.get(cv2.CAP_PROP_FPS)
        
        out = cv2.VideoWriter(OUTPUT_PATH, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            processed = process_frame(frame, model_yolo, clf_model, retriever_assets, DEVICE)
            out.write(processed)
            cv2.imshow("Processing Video", processed)
            
            if cv2.waitKey(1) & 0xFF == 27: break
            
        cap.release()
        out.release()
        print(f"Saved video to {OUTPUT_PATH}")

    cv2.destroyAllWindows()