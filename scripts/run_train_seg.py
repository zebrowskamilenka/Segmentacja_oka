from ultralytics import YOLO

# Załadowanie modelu segmentacyjnego
model = YOLO("yolo11n-seg.pt")

# Trening segmentacji
results = model.train(
    data="dataset/data.yaml",
    epochs=50,
    imgsz=512,
    device="cpu"
)

print("Trening segmentacji zakończony.")