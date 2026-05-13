import os
import json

# Ścieżki
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

JSON_DIR = os.path.join(BASE_DIR, "dataset", "json")

TRAIN_LABELS_DIR = os.path.join(BASE_DIR, "dataset", "train", "labels")
VAL_LABELS_DIR = os.path.join(BASE_DIR, "dataset", "val", "labels")

TRAIN_IMAGES_DIR = os.path.join(BASE_DIR, "dataset", "train", "images")
VAL_IMAGES_DIR = os.path.join(BASE_DIR, "dataset", "val", "images")

os.makedirs(TRAIN_LABELS_DIR, exist_ok=True)
os.makedirs(VAL_LABELS_DIR, exist_ok=True)

# Mapowanie klas
CLASS_MAP = {
    "pupil": 0,
    "pupiil": 0,
    "iris": 1,
    "sclera": 2,
    # background i skin 
}


# Normalizacja punktów do zakresu 0-1
def normalize_point(x, y, img_w, img_h):

    x_norm = x / img_w
    y_norm = y / img_h

    return x_norm, y_norm


# Przejście po wszystkich jsonach
for filename in os.listdir(JSON_DIR):

    # Pomijamy nie-jsony
    if not filename.endswith(".json"):
        continue

    json_path = os.path.join(JSON_DIR, filename)

    # Wczytanie jsona
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Pobranie obiektów
    shapes = data.get("shapes", [])

    # Rozmiar obrazu
    img_h = data.get("imageHeight")
    img_w = data.get("imageWidth")

    # Linie do zapisania
    yolo_lines = []

    # Iteracja po obiektach
    for shape in shapes:

        label = shape.get("label")
        points = shape.get("points", [])

        # Nieznana klasa
        if label not in CLASS_MAP:
            print(f"Pominięto etykietę '{label}' w pliku {filename}")
            continue

        # Brak punktów
        if not points:
            print(f"Pominięto pusty shape w pliku {filename}")
            continue

        # Za mało punktów
        if len(points) < 3:
            print(f"Za mało punktów dla '{label}' w pliku {filename}")
            continue

        class_id = CLASS_MAP[label]

        # Lista punktów polygonu
        polygon_points = []

        # Normalizacja punktów
        for point in points:

            x = point[0]
            y = point[1]

            x_norm, y_norm = normalize_point(x, y, img_w, img_h)

            polygon_points.append(f"{x_norm:.6f}")
            polygon_points.append(f"{y_norm:.6f}")

        # Format YOLO segmentation
        line = f"{class_id} " + " ".join(polygon_points)

        yolo_lines.append(line)

    # Nazwa pliku txt
    out_name = os.path.splitext(filename)[0] + ".txt"

    # Szukanie obrazu w train/val
    image_name = os.path.splitext(filename)[0] + ".png"

    train_image_path = os.path.join(TRAIN_IMAGES_DIR, image_name)
    val_image_path = os.path.join(VAL_IMAGES_DIR, image_name)

    # Gdzie zapisać label
    if os.path.exists(train_image_path):

        out_path = os.path.join(TRAIN_LABELS_DIR, out_name)

    elif os.path.exists(val_image_path):

        out_path = os.path.join(VAL_LABELS_DIR, out_name)

    else:

        print(f"Nie znaleziono obrazu dla {filename}")
        continue

    # Zapis txt
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(yolo_lines))

    print(f"Zapisano: {out_path}")

print("Gotowe.")