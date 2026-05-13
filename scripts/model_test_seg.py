from ultralytics import YOLO

# Wczytanie modelu
model = YOLO("runs/segment/train/weights/best.pt")

# Predykcja
results = model.predict(
    source="dataset/val/images",
    save=True,
    imgsz=512,
    conf=0.25
)

print("Predykcja zakończona.")