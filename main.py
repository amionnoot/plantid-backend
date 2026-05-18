"""FastAPI entrypoint for crop/weed detection."""

import uuid
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from model import run_inference

IMAGES_DIR = Path("images")
IMAGES_DIR.mkdir(exist_ok=True)

app = FastAPI(
    title="Plant.ID Crop/Weed Detection",
    description="YOLO11m fine-tuned on CropAndWeed (CropOrWeed2 variant).",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/images", StaticFiles(directory=str(IMAGES_DIR)), name="images")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/analyze/")
async def analyze(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="Could not decode image.")

    annotated, detections = run_inference(image)

    fname = f"{uuid.uuid4().hex}.jpg"
    out_path = IMAGES_DIR / fname
    cv2.imwrite(str(out_path), annotated)

    return JSONResponse({
        "filename": file.filename,
        "detections": detections,
        "image_url": f"/images/{fname}",
    })
