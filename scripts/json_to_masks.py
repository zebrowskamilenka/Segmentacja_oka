import os
import json
import base64
import io
from PIL import Image
import numpy as np
from labelme import utils

# Ścieżki do folderów
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LABEL_DIR = os.path.join(BASE_DIR, "dataset", "json")
MASK_DIR = os.path.join(BASE_DIR, "dataset", "masks")
IMAGE_DIR = os.path.join(BASE_DIR, "dataset", "images")

os.makedirs(MASK_DIR, exist_ok=True)

# Mapowanie klas
CLASS_MAP = {
    "_background_": 0,
    "pupil": 1,
    "pupiil": 1,
    "iris": 2,
    "sclera": 3,
    "skin": 4,
}

def load_image_from_json(data, json_path):
    image_data = data.get("imageData")

    if image_data:
        image_bytes = base64.b64decode(image_data)
        return np.array(Image.open(io.BytesIO(image_bytes)).convert("RGB"))

    image_path = data.get("imagePath")
    if not image_path:
        raise ValueError(f"Brak imageData i imagePath w pliku: {json_path}")

    abs_image_path = os.path.join(IMAGE_DIR, os.path.basename(image_path))

    if not os.path.exists(abs_image_path):
        raise FileNotFoundError(f"Nie znaleziono obrazu: {abs_image_path}")

    return np.array(Image.open(abs_image_path).convert("RGB"))

for filename in os.listdir(LABEL_DIR):
    if not filename.endswith(".json"):
        continue

    json_path = os.path.join(LABEL_DIR, filename)

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    img = load_image_from_json(data, json_path)
    shapes = data.get("shapes", [])

    if not shapes:
        print(f"Pominięto {filename} - brak shapes")
        continue

    labels_in_file = set(shape["label"] for shape in shapes)
    print(f"{filename} -> etykiety: {labels_in_file}")

    unknown = sorted(labels_in_file - set(CLASS_MAP.keys()))
    if unknown:
        raise ValueError(
            f"W pliku {filename} są etykiety, których nie ma w CLASS_MAP: {unknown}"
        )

    mask, _ = utils.shapes_to_label(img.shape, shapes, CLASS_MAP)

    print(f"{filename} -> unikalne wartości maski: {np.unique(mask)}")

    # Finalna maska do datasetu
    out_name = os.path.splitext(filename)[0] + ".png"
    out_path = os.path.join(MASK_DIR, out_name)
    Image.fromarray(mask.astype(np.uint8)).save(out_path)
    print(f"Zapisano maskę: {out_path}")

    # Podgląd dla człowieka
    preview = (mask * 60).clip(0, 255).astype(np.uint8)
    preview_name = os.path.splitext(filename)[0] + "_preview.png"
    preview_path = os.path.join(MASK_DIR, preview_name)
    Image.fromarray(preview).save(preview_path)
    print(f"Zapisano podgląd: {preview_path}")

print("Gotowe.")