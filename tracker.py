"""
Lightweight tracking of a confirmed animal, so classification does not
have to run again on every frame.
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
        # CSRT is more accurate, KCF faster - pick based on available compute
        self.tracker = cv2.legacy.TrackerKCF_create() if hasattr(cv2, "legacy") else cv2.TrackerKCF_create()
        self.tracker.init(frame, (x, y, w, h))
        self.active = True
        self.frames_since_start = 0
        self.label = label

    def update(self, frame):
        """Return (ok, box). With ok=False the tracking is lost."""
        if not self.active:
            return False, None
        if frame is None or frame.size == 0:
            self.stop()
            return False, None
        if frame.ndim >= 2 and (frame.shape[0] == 0 or frame.shape[1] == 0):
            self.stop()
            return False, None

        try:
            ok, box = self.tracker.update(frame)
        except cv2.error:
            self.stop()
            return False, None

        self.frames_since_start += 1
        if not ok:
            self.active = False
            self.tracker = None
        return ok, tuple(int(v) for v in box) if ok else None

    def needs_reclassification(self):
        return self.frames_since_start >= self.reclassify_interval_frames

    def stop(self):
        self.active = False
        self.tracker = None
