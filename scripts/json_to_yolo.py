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

