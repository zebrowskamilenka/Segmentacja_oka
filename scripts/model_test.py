from ultralytics import YOLO
# wczytanie naszego modelu
model = YOLO('runs/detect/train-2/weights/best.pt')
# testowanie modelu na nowych danych
print("Testowanie całego folderu...")
# results = model('dataset/img_t/z1.jpg')
model.predict('dataset/img_t', save=True)
