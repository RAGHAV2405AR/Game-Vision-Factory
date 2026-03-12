import torch
from ultralytics import YOLO

def train_model(data_yaml, epochs):
    device = "0" if torch.cuda.is_available() else "cpu"
    print("Training on device:", "GPU" if device == "0" else "CPU")
    model = YOLO("yolov8n.pt")
    model.train(data=data_yaml, epochs=epochs, device=device)
