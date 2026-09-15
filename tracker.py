"""
Leichtgewichtiges Tracking eines bestätigten Tiers, um nicht bei
jedem Frame erneut klassifizieren zu müssen.
"""

import cv2


class AnimalTracker:
    def __init__(self, reclassify_interval_frames=60):
        self.tracker = None
        self.active = False
        self.frames_since_start = 0
        self.reclassify_interval_frames = reclassify_interval_frames
        self.label = None

    def start(self, frame, box, label):
        x, y, w, h = box
        # CSRT ist genauer, KCF schneller - je nach Rechenleistung wählen
        self.tracker = cv2.legacy.TrackerKCF_create() if hasattr(cv2, "legacy") else cv2.TrackerKCF_create()
        self.tracker.init(frame, (x, y, w, h))
        self.active = True
        self.frames_since_start = 0
        self.label = label

    def update(self, frame):
        """Gibt (ok, box) zurück. Bei ok=False ist das Tracking verloren."""
        if not self.active:
            return False, None
        ok, box = self.tracker.update(frame)
        self.frames_since_start += 1
        if not ok:
            self.active = False
        return ok, tuple(int(v) for v in box) if ok else None

    def needs_reclassification(self):
        return self.frames_since_start >= self.reclassify_interval_frames

    def stop(self):
        self.active = False
        self.tracker = None
