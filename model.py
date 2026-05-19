"""YOLO11m crop/weed inference.

Loads the fine-tuned model once at import time and exposes run_inference()
which takes a BGR OpenCV image and returns the annotated image plus a list
of detections. Uses Ultralytics' built-in annotator so the output looks
identical to `yolo detect predict ...`.
"""

import numpy as np
from ultralytics import YOLO

MODEL_PATH = "weights/best.pt"
CONF_THRESHOLD = 0.35
IOU_THRESHOLD = 0.5
IMGSZ = 640

model = YOLO(MODEL_PATH)


def run_inference(image: np.ndarray):
    results = model(image, conf=CONF_THRESHOLD, iou=IOU_THRESHOLD, imgsz=IMGSZ, verbose=False)[0]

    boxes = []
    for r in results.boxes:
        x1, y1, x2, y2 = map(int, r.xyxy[0].tolist())
        boxes.append({
            "label": model.names[int(r.cls[0])],
            "confidence": round(float(r.conf[0]), 3),
            "box": [x1, y1, x2, y2],
        })

    # Ultralytics' built-in renderer: same visual style as `yolo detect predict`
    # auto-scales line width and font size to image resolution.
    annotated = results.plot(line_width=None, font_size=None, conf=True, labels=True)

    return annotated, boxes
