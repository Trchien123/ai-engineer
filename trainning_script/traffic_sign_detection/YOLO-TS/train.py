from ultralytics import YOLO
import torch

# Security fix for PyTorch 2.6+
try:
    from ultralytics.nn.tasks import DetectionModel
    torch.serialization.add_safe_globals([DetectionModel])
except ImportError:
    pass

if __name__ == '__main__':
    # 1. Load the architecture configuration
    model = YOLO("./YOLO-TS.yaml") 

    # 2. Train using your custom data yaml
    model.train(
        data="../data/damaged-sign3/final_dataset_split_4/data.yaml", 
        epochs=100, 
        batch=16,       # Adjust based on your GPU memory
        imgsz=1024,     # Better for small signs
        device='cpu',       # Use 'cpu' if no GPU
        project="YOLO-TS-AUS",
        name="sign_detection"
    )