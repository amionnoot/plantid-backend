"""YOLO11m crop/weed inference.

Loads the fine-tuned model once at import time and exposes run_inference()
which takes a BGR OpenCV image and returns the annotated image plus a list
of detections.
"""

import cv2
import numpy as np
from ultralytics import YOLO

MODEL_PATH = "weights/best.pt"
CONF_THRESHOLD = 0.35
IOU_THRESHOLD = 0.5
IMGSZ = 640

# BGR colors per class index (0 = crop, 1 = weed)
CLASS_COLORS = {
    0: (255, 100, 0),    # crop -> blue
    1: (0, 200, 255),    # weed -> cyan/yellow
}
DEFAULT_COLOR = (0, 255, 0)

model = YOLO(MODEL_PATH)


def run_inference(image: np.ndarray):
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = model(rgb, conf=CONF_THRESHOLD, iou=IOU_THRESHOLD, imgsz=IMGSZ, verbose=False)[0]

    boxes = []
    for r in results.boxes:
        x1, y1, x2, y2 = map(int, r.xyxy[0].tolist())
        conf = float(r.conf[0])
        cls = int(r.cls[0])
        label = model.names[cls]
        color = CLASS_COLORS.get(cls, DEFAULT_COLOR)

        boxes.append({
            "label": label,
            "confidence": round(conf, 3),
            "box": [x1, y1, x2, y2],
        })

        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        text = f"{label} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(image, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(image, text, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    return image, boxes
