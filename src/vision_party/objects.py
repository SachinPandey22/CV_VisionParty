"""Object detection with a pretrained YOLOv8n (COCO), used by Bring to Camera.

The model is loaded lazily on a background thread (`preload`), so starting a
match never freezes the video. Until it's ready, `detect` returns nothing.
Detection runs only while an object round is active (see main.py), so the
pose-only games keep their full frame rate.
"""

import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import config

# Weights are downloaded by Ultralytics on first use into this folder (~6 MB).
MODEL_PATH = Path(__file__).resolve().parents[2] / "models" / config.OBJECT_MODEL


@dataclass
class Detection:
    name: str                 # COCO class name, e.g. "bottle"
    confidence: float
    box: tuple[int, int, int, int]  # x0, y0, x1, y1 in raw-frame pixels
    area_frac: float          # box area / frame area: how close the object is


class ObjectDetector:
    def __init__(self, classes: tuple[str, ...] = config.OBJECT_CLASSES):
        self.classes = classes
        self._model = None
        self._class_ids: list[int] = []
        self._thread: threading.Thread | None = None
        self.error: str | None = None

    @property
    def ready(self) -> bool:
        return self._model is not None

    def preload(self) -> None:
        """Start loading the model in the background (no-op if already started)."""
        if self._thread is None and not self.ready:
            self._thread = threading.Thread(target=self._load, daemon=True)
            self._thread.start()

    def load(self) -> None:
        """Load synchronously (used by tests and the evaluation script)."""
        if not self.ready:
            self._load()

    def _load(self) -> None:
        try:
            from ultralytics import YOLO   # imported here: slow import, only needed for object rounds
            MODEL_PATH.parent.mkdir(exist_ok=True)
            model = YOLO(str(MODEL_PATH))
            name_to_id = {n: i for i, n in model.names.items()}
            self._class_ids = [name_to_id[c] for c in self.classes]
            # One warm-up run so the first real frame isn't slow.
            model.predict(np.zeros((640, 640, 3), np.uint8), verbose=False)
            self._model = model
        except Exception as exc:  # missing package, no network for first download, ...
            self.error = f"{type(exc).__name__}: {exc}"
            print(f"[objects] could not load YOLO: {self.error}")

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """Detect the supported objects in a raw BGR frame, most confident first."""
        if not self.ready:
            return []
        h, w = frame.shape[:2]
        result = self._model.predict(frame, classes=self._class_ids, conf=config.OBJECT_MIN_CONFIDENCE,
                                     imgsz=config.OBJECT_IMG_SIZE, verbose=False)[0]
        out = []
        for box, cls, conf in zip(result.boxes.xyxy.tolist(), result.boxes.cls.tolist(),
                                  result.boxes.conf.tolist()):
            x0, y0, x1, y1 = (int(v) for v in box)
            area = max(0, x1 - x0) * max(0, y1 - y0) / (w * h)
            out.append(Detection(self._model.names[int(cls)], float(conf), (x0, y0, x1, y1), area))
        return sorted(out, key=lambda d: -d.confidence)
