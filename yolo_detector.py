"""YOLO detection via Ultralytics, interface-compatible with MotionDetector."""

import cv2
import torch
from ultralytics import YOLO


class YoloDetector:
    def __init__(self, model_path="yolo11n.pt", conf=0.4, classes=None,
                 device=None, imgsz=640, warmup_frames=1, verbose=False):
        self.device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = YOLO(model_path)
        self.conf = conf
        self.classes = classes or None
        self.imgsz = imgsz
        self.verbose = verbose
        self.warmup_frames = warmup_frames   # Shim: wie MotionDetector
        self._frames_seen = 0

    # --- Status-Shims, damit run() identisch funktioniert ---
    @property
    def frames_seen(self):
        return self._frames_seen

    @property
    def warming_up(self):
        return self._frames_seen < self.warmup_frames

    def _infer(self, frame):
        kwargs = dict(source=frame, conf=self.conf, device=self.device,
                      imgsz=self.imgsz, verbose=self.verbose)
        if self.classes is not None:
            kwargs["classes"] = self.classes
        return self.model.predict(**kwargs)

    def detect(self, frame):
        """Boxes (x, y, w, h) in Originalkoordinaten."""
        results = self._infer(frame)
        self._frames_seen += 1
        if self.warming_up:
            return []
        boxes = []
        for r in results:
            for x1, y1, x2, y2 in r.boxes.xyxy.cpu().numpy().astype(int):
                boxes.append((int(x1), int(y1), int(x2 - x1), int(y2 - y1)))
        return boxes

    def overlay_motion_mask(self, frame, alpha=1.0):
        """Alias für Interface-Kompatibilität; zeichnet Boxen statt Maske."""
        display = frame.copy()
        for x, y, w, h in self.detect(frame):
            cv2.rectangle(display, (x, y), (x + w, y + h), (0, 220, 0), 2)
        return display