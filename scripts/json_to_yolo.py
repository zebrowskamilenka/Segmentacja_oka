import os
import json

#ścieżki 
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
JSON_DIR = os.path.join(BASE_DIR, "dataset", "json")
YOLO_DIR = os.path.join (BASE_DIR, "dataset", "labels")

os.makedirs(YOLO_DIR, exist_ok=True)

#Mapowanie klas
CLASS_MAP={
    "pupil": 1,
    "pupiil":1,
    "iris": 2,
    "sclera": 3,
    "skin": 4,

}
#wyzanaczam bounding box z punktów wielokąta
def polygon_to_bbox(points):
    xs = [p[0] for p in points] #wszytskie współrzedne x
    ys = [p[1] for p in points] #wszystkie y

    x_min = min(xs) #lewa krawędź
    y_min = min(ys) #prawa krwedz
    x_max = max(xs) #prawa krawedz  
    y_max = max(ys) # dolna 

    return x_min, y_min, x_max, y_max

#bbox na format YOLO
def bbox_to_yolo(x_min, y_min, x_max, y_max, img_w, img_h):
    # środek prostokąta
    x_center = ((x_min + x_max) / 2) / img_w
    y_center = ((y_min + y_max) / 2) / img_h

    # szerokość i wysokość prostokąta
    width = (x_max - x_min) / img_w
    height = (y_max - y_min) / img_h

    return x_center, y_center, width, height

# Przejście po wszystkich plikach w folderze json
for filename in os.listdir(JSON_DIR):

    # Pomijamy pliki, które nie są jsonami
    if not filename.endswith(".json"):
        continue

    # Pełna ścieżka do pliku
    json_path = os.path.join(JSON_DIR, filename)

    # Otwórz i wczytaj plik json
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Pobierz listę obiektów zaznaczonych na obrazie
    shapes = data.get("shapes", [])

    # Pobierz szerokość i wysokość obrazu
    img_h = data.get("imageHeight")
    img_w = data.get("imageWidth")
    # Lista linii do zapisania w pliku txt
    yolo_lines = []
     # Przejście po wszystkich zaznaczonych obiektach
    for shape in shapes:
        label = shape.get("label")          # nazwa klasy
        points = shape.get("points", [])    # punkty obiektu

        # Jeśli etykieta nie jest znana, pomiń
        if label not in CLASS_MAP:
            print(f"Pominięto etykietę '{label}' w pliku {filename}")
            continue

        # Jeśli nie ma punktów, pomiń
        if not points:
            print(f"Pominięto pusty shape w pliku {filename}")
            continue

        # Numer klasy
        class_id = CLASS_MAP[label]

        # Zamiana punktów na bounding box
        x_min, y_min, x_max, y_max = polygon_to_bbox(points)

        # Zamiana bboxa na format YOLO
        x_center, y_center, width, height = bbox_to_yolo(
            x_min, y_min, x_max, y_max, img_w, img_h
        )

        # Stworzenie jednej linii tekstu do pliku txt
        line = f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
        yolo_lines.append(line)

    # Nazwa pliku wyjściowego, np. oko1.txt
    out_name = os.path.splitext(filename)[0] + ".txt"
    out_path = os.path.join(YOLO_DIR, out_name)

    # Zapis linii do pliku txt
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(yolo_lines))

    print(f"Zapisano: {out_path}")

print("Gotowe.")