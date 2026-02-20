from ultralytics import YOLO

def train_model(data_yaml: str, epochs: int):
    model = YOLO("yolov8n.pt")
    model.train(
        data=data_yaml,
        epochs=epochs,
        device="cpu"
    )
