"""
Traditionelle Bewegungserkennung per Background Subtraction (MOG2).
Liefert Bounding-Boxes der bewegten Bereiche, gefiltert nach Mindestfläche.
"""

import cv2


def to_grayscale(frame):
    if frame is None:
        return frame
    if frame.ndim == 2:
        return frame
    if frame.ndim == 3 and frame.shape[2] == 1:
        return frame[:, :, 0]
    if frame.ndim == 3 and frame.shape[2] == 3:
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if frame.ndim == 3 and frame.shape[2] == 4:
        return cv2.cvtColor(frame, cv2.COLOR_BGRA2GRAY)
    return frame


class MotionDetector:
    def __init__(
        self, min_area=1200, max_area=50000, downscale_width=320, history=500, var_threshold=10
    ):
        self.min_area = min_area
        self.max_area = max_area
        self.downscale_width = downscale_width
        self.last_mask = None
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=history, varThreshold=var_threshold, detectShadows=False
        )

    def _scale_factor(self, frame):
        h, w = frame.shape[:2]
        if w <= self.downscale_width:
            return 1.0, frame
        factor = self.downscale_width / w
        small = cv2.resize(frame, (self.downscale_width, int(h * factor)))
        return factor, small

    def detect(self, frame):
        """
        Gibt eine Liste von Bounding-Boxes (x, y, w, h) im Koordinatensystem
        des Original-Frames zurück, gefiltert nach Mindestfläche.
        """
        grayscale = to_grayscale(frame)
        factor, small = self._scale_factor(grayscale)
        mask = self.bg_subtractor.apply(small)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=1)
        self.last_mask = mask if factor == 1.0 else cv2.resize(
            mask, (grayscale.shape[1], grayscale.shape[0]), interpolation=cv2.INTER_NEAREST
        )

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        boxes = []
        for c in contours:
            area = cv2.contourArea(c)
            if area < self.min_area * (factor ** 2) or area > self.max_area * (factor ** 2):
                continue
            x, y, w, h = cv2.boundingRect(c)
            boxes.append((int(x / factor), int(y / factor), int(w / factor), int(h / factor)))

        return boxes

    def overlay_motion_mask(self, frame, alpha=1.0):
        """Rote Bewegungsmaske auf einer BGR-Kopie; Eingabe bleibt unverändert."""
        if frame.ndim == 3 and frame.shape[2] == 3:
            display = frame.copy()
        else:
            display = cv2.cvtColor(to_grayscale(frame), cv2.COLOR_GRAY2BGR)
        if self.last_mask is None:
            return display

        active = self.last_mask > 0
        overlay = display.copy()
        overlay[active] = (0, 0, 255)
        return cv2.addWeighted(overlay, alpha, display, 1.0 - alpha, 0)
