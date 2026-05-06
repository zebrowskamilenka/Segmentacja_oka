from ultralytics import YOLO
# Załadowanie modelu nano
model = YOLO('yolo11n.pt')
# Trenowanie modelu na własnym zbiorze danych
results = model.train(data='dataset/data.yaml', epochs=50, imgsz=512, device='cpu')