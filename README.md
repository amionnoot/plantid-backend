# Plant.ID Backend — Crop/Weed Detection

REST-API zur Detektion von Nutzpflanzen (crop) und Unkraut (weed) auf landwirtschaftlichen Bildern. Basiert auf einem auf dem CropAndWeed-Datensatz fine-getuneten **YOLO11m**-Modell.

## Performance (Test-Set, 771 Bilder, 7621 Instanzen)

| Metrik          | Wert  |
|-----------------|-------|
| mAP@0.5         | 0.829 |
| mAP@0.5:0.95    | 0.596 |
| crop mAP@0.5    | 0.931 |
| weed mAP@0.5    | 0.727 |
| Inferenz/Bild   | ~14 ms (GTX 1660 Super) |

## Setup (lokal)

```bash
pip install -r requirements.txt

# Modell-Gewichte ablegen
# Datei "best.pt" nach ./weights/best.pt kopieren

uvicorn main:app --reload
```

Server läuft danach auf `http://localhost:8000`. Swagger-UI unter `http://localhost:8000/docs`.

## Setup (Docker)

```bash
docker compose up --build
```

## API

### `POST /analyze/`

Multipart-Upload eines Bildes (`file`-Feld). Antwort:

```json
{
  "filename": "input.jpg",
  "detections": [
    {"label": "crop", "confidence": 0.93, "box": [x1, y1, x2, y2]},
    {"label": "weed", "confidence": 0.81, "box": [x1, y1, x2, y2]}
  ],
  "image_url": "/images/<uuid>.jpg"
}
```

Das annotierte Bild ist direkt über `GET /images/<uuid>.jpg` abrufbar.

### `GET /health`

Healthcheck — gibt `{"status": "ok"}` zurück.

## Inferenz-Parameter

Defaults in [model.py](model.py):

- `CONF_THRESHOLD = 0.35` — empirisch ermittelt (F1-Optimum)
- `IOU_THRESHOLD = 0.5` — reduziert Mehrfach-Detektionen großer Pflanzen
- `IMGSZ = 640` — Trainings-Auflösung

## Modell-Details

- **Architektur**: YOLO11m (20.1 M Parameter, 68.2 GFLOPs)
- **Vortrainiert**: COCO
- **Fine-tuning**: 100 Epochen, CropAndWeed (CropOrWeed2-Variante, 2 Klassen)
- **Split**: 80/10/10 train/val/test, Seed=42
- **Hardware**: NVIDIA GTX 1660 Super, 6 GB VRAM

## Lizenz und Hinweise

Das zugrundeliegende **CropAndWeed-Datenset** ist ausschließlich für **akademische und nicht-kommerzielle Nutzung** lizenziert (siehe Steininger et al., WACV 2023). Folglich ist auch dieses Modell nicht für kommerzielle Anwendungen freigegeben.

Bei Verwendung bitte zitieren:

```bibtex
@InProceedings{Steininger2023CropAndWeedDataset,
  author    = {Steininger, Daniel and Trondl, Andreas and Croonen, Gerardus and Simon, Julia and Widhalm, Verena},
  title     = {The CropAndWeed Dataset: a Multi-Modal Learning Approach for Efficient Crop and Weed Manipulation},
  booktitle = {WACV},
  year      = {2023}
}
```

## Bekannte Limitierungen

- Blühende Pflanzen werden schlechter erkannt als vegetative — Trainingsdaten enthalten überwiegend Blattstadien.
- Sehr dichte Mischszenen liefern teilweise überlappende Detektionen.
- Modell ist domain-spezifisch (europäische Ackerflächen).
