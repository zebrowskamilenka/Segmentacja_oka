import os  #operacje na plikach
import json #czytanie json
import base64 #dekodowanie z json
from PIL import Image # zapis do PNG
import numpy as np #macierze
from labelme import utils 
import io

#ścieżki do folderów
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) #znajdz folder główny projektu   
LABEL_DIR = os.path.join(BASE_DIR, "dataset", "labels") #skad biore jsony
MASK_DIR =  os.path.join(BASE_DIR, "dataset", "masks") #gdzie dać maski
IMAGE_DIR = os.path.join(BASE_DIR, "dataset", "images") #gdzie obrazy

os.makedirs(MASK_DIR, exist_ok=True)

CLASS_MAP = {
    "_background_ ":0,
    "pupil": 1,
    "pupiil":1,
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
    unknown = sorted(labels_in_file - set(CLASS_MAP.keys()))
    if unknown:
        raise ValueError(
            f"W pliku {filename} są etykiety, których nie ma w CLASS_MAP: {unknown}"
        )

    mask, _ = utils.shapes_to_label(
        img.shape,
        shapes,
        CLASS_MAP
    )
    print(filename, np.unique(mask))

    out_name = os.path.splitext(filename)[0] + "_mask.png"
    out_path = os.path.join(MASK_DIR, out_name)

    Image.fromarray(mask.astype(np.uint8)).save(out_path)
    print(f"Zapisano: {out_path}")

    preview = (mask * 60).clip(0, 255).astype(np.uint8)
    preview_name = os.path.splitext(filename)[0] + "_preview.png"
    preview_path = os.path.join(MASK_DIR, preview_name)
    Image.fromarray(preview).save(preview_path)

    print(f"Zapisano podgląd: {preview_path}")

print("Gotowe.")