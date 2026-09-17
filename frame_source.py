"""
Austauschbare Bildquellen für die Pipeline.

Aktuell: Videodatei oder Webcam (für Entwicklung ohne Hardware).
Später: eine CsiFrameSource-Klasse mit derselben Schnittstelle,
die auf dem RV1106/Duo-S das CSI-Device ausliest (z.B. über
V4L2 / /dev/videoX). Der Rest der Pipeline muss dafür nicht
angepasst werden, solange get_frame() weiterhin ein (ok, frame)
Tupel liefert.
"""

from abc import ABC, abstractmethod
import random
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


class FrameSource(ABC):
    """Gemeinsame Schnittstelle für alle Bildquellen."""

    @abstractmethod
    def get_frame(self):
        """Liefert (ok: bool, frame: np.ndarray | None)."""
        raise NotImplementedError

    def release(self):
        pass


class VideoFileSource(FrameSource):
    """Liest Frames aus einer Videodatei (z.B. Wildlife-Datensatz-Clip)."""

    def __init__(self, path: str, loop: bool = True):
        self.path = path
        self.loop = loop
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            raise FileNotFoundError(f"Videodatei konnte nicht geöffnet werden: {path}")

    def get_frame(self):
        ok, frame = self.cap.read()
        if not ok and self.loop:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = self.cap.read()
        if ok and frame is not None:
            frame = to_grayscale(frame)
        return ok, frame

    def release(self):
        self.cap.release()


class WebcamSource(FrameSource):
    """Liest Frames von einer lokalen Webcam (Index, meist 0)."""

    def __init__(self, index: int = 0):
        self.cap = cv2.VideoCapture(index)
        if not self.cap.isOpened():
            raise RuntimeError(f"Webcam mit Index {index} konnte nicht geöffnet werden")

    def get_frame(self):
        ok, frame = self.cap.read()
        if ok and frame is not None:
            frame = to_grayscale(frame)
        return ok, frame

    def release(self):
        self.cap.release()


class SyntheticMotionSource(FrameSource):
    """
    Erzeugt synthetische Wildlife-Frames mit einem transparenten Sprite,
    einem langsam wechselnden Tages-/Nachthintergrund und einer Bewegung,
    die das Bild vollständig verlassen kann.
    """

    def __init__(self, width=640, height=480, speed=6, sprite_path=None):
        import numpy as np

        self.width = width
        self.height = height
        self.speed = max(1, speed)
        self.rng = random.Random()
        self.frame_index = 0
        self.sprite = self._load_sprite(sprite_path)
        self.sprite_h, self.sprite_w = self.sprite.shape[:2]
        self.x = -float(self.sprite_w)
        self.y = self.height * 0.62
        self.direction_x = 1
        self.direction_y = 1
        self.vx = max(1.0, float(self.speed) * 0.8)
        self.vy = max(0.3, float(self.speed) * 0.18)
        self._respawn_gap = 0

    @staticmethod
    def _default_sprite():
        import numpy as np

        sprite = np.zeros((150, 190, 4), dtype=np.uint8)
        body = (48, 72, 70, 255)
        detail = (30, 44, 43, 255)
        cv2.ellipse(sprite, (92, 78), (62, 34), 0, 0, 360, body, -1)
        cv2.ellipse(sprite, (150, 57), (28, 23), -15, 0, 360, body, -1)
        cv2.ellipse(sprite, (166, 49), (10, 7), -15, 0, 360, detail, -1)
        cv2.line(sprite, (55, 101), (48, 142), detail, 9)
        cv2.line(sprite, (93, 105), (89, 146), detail, 9)
        cv2.line(sprite, (126, 101), (134, 143), detail, 9)
        cv2.line(sprite, (47, 71), (18, 48), detail, 7)
        cv2.line(sprite, (155, 39), (147, 13), detail, 5)
        cv2.line(sprite, (170, 39), (179, 16), detail, 5)
        cv2.circle(sprite, (158, 52), 3, (225, 210, 150, 255), -1)
        return sprite

    def _load_sprite(self, sprite_path):
        import numpy as np

        if sprite_path:
            sprite = cv2.imread(sprite_path, cv2.IMREAD_UNCHANGED)
            if sprite is None:
                raise FileNotFoundError(f"Sprite konnte nicht geöffnet werden: {sprite_path}")
            if sprite.ndim == 2:
                sprite = cv2.cvtColor(sprite, cv2.COLOR_GRAY2BGRA)
            elif sprite.shape[2] == 3:
                alpha = 255 * (np.any(sprite > 8, axis=2)).astype("uint8")
                sprite = np.dstack((sprite, alpha))
            return sprite
        return self._default_sprite()

    def _background(self):
        import numpy as np

        phase = (self.frame_index % 1800) / 1800.0
        daylight = (np.sin(phase * 2 * np.pi - np.pi / 2) + 1.0) / 2.0
        daylight = 0.12 + 0.88 * daylight
        top = np.array([18, 26, 32], dtype=np.float32)
        bottom = np.array([42, 57, 45], dtype=np.float32)
        if daylight > 0.5:
            top += np.array([35, 34, 22], dtype=np.float32) * daylight
            bottom += np.array([30, 36, 18], dtype=np.float32) * daylight
        vertical = np.linspace(0, 1, self.height, dtype=np.float32)[:, None, None]
        frame = top * (1 - vertical) + bottom * vertical
        frame = np.broadcast_to(frame, (self.height, self.width, 3)).copy()

        # Low-resolution fields create independent, slowly drifting light zones.
        grid_y, grid_x = np.mgrid[0:6, 0:8]
        noise = 0.9 + 0.055 * np.sin(
            self.frame_index / 95.0 + grid_x * 0.8 + grid_y * 1.3
        )
        noise += 0.025 * np.sin(self.frame_index / 170.0 + grid_x * 1.7 - grid_y)
        noise = cv2.resize(noise, (self.width, self.height), interpolation=cv2.INTER_CUBIC)
        frame *= noise[:, :, None] * (0.75 + 0.25 * daylight)
        return np.clip(frame, 0, 255).astype(np.uint8)

    def _update_position(self):
        if self._respawn_gap:
            self._respawn_gap -= 1
            return

        self.vx = min(max(self.vx + self.rng.uniform(-0.25, 0.35), 0.5), self.speed * 1.5)
        self.vy = min(max(self.vy + self.rng.uniform(-0.12, 0.12), 0.15), self.speed * 0.5)
        self.y += self.direction_y * self.vy
        if self.y < self.height * 0.12 or self.y > self.height * 0.78:
            self.direction_y *= -1
        self.x += self.direction_x * self.vx

        if self.direction_x > 0 and self.x > self.width + self.sprite_w * 0.4:
            self.direction_x = -1
            self.x = self.width + self.sprite_w * 0.4
            self._respawn_gap = self.rng.randint(8, 35)
        elif self.direction_x < 0 and self.x + self.sprite_w < -self.sprite_w * 0.4:
            self.direction_x = 1
            self.x = -self.sprite_w * 1.4
            self._respawn_gap = self.rng.randint(8, 35)

    def _sprite_frame(self):
        import numpy as np

        depth = np.clip(self.y / max(1, self.height), 0, 1)
        scale = 0.42 + depth * 0.78
        angle = self.direction_x * (self.vy * 1.8) + np.sin(self.frame_index / 9.0) * 2.5
        sprite = self.sprite
        if self.direction_x < 0:
            sprite = cv2.flip(sprite, 1)
        new_w = max(2, int(self.sprite_w * scale))
        new_h = max(2, int(self.sprite_h * scale))
        sprite = cv2.resize(sprite, (new_w, new_h), interpolation=cv2.INTER_AREA)
        matrix = cv2.getRotationMatrix2D((new_w / 2, new_h / 2), angle, 1.0)
        bound_w = max(2, int(abs(matrix[0, 0]) * new_w + abs(matrix[0, 1]) * new_h))
        bound_h = max(2, int(abs(matrix[0, 0]) * new_h + abs(matrix[0, 1]) * new_w))
        matrix[0, 2] += bound_w / 2 - new_w / 2
        matrix[1, 2] += bound_h / 2 - new_h / 2
        return cv2.warpAffine(sprite, matrix, (bound_w, bound_h), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0, 0))

    def _composite(self, frame, sprite):
        import numpy as np

        sh, sw = sprite.shape[:2]
        x0, y0 = int(self.x), int(self.y - sh / 2)
        x1, y1 = x0 + sw, y0 + sh
        clip_x0, clip_y0 = max(0, x0), max(0, y0)
        clip_x1, clip_y1 = min(self.width, x1), min(self.height, y1)
        if clip_x0 >= clip_x1 or clip_y0 >= clip_y1:
            return
        sx0, sy0 = clip_x0 - x0, clip_y0 - y0
        sx1, sy1 = sx0 + clip_x1 - clip_x0, sy0 + clip_y1 - clip_y0
        overlay = sprite[sy0:sy1, sx0:sx1]
        alpha = overlay[:, :, 3:4].astype(np.float32) / 255.0
        target = frame[clip_y0:clip_y1, clip_x0:clip_x1].astype(np.float32)
        frame[clip_y0:clip_y1, clip_x0:clip_x1] = (overlay[:, :, :3] * alpha + target * (1 - alpha)).astype(np.uint8)

    def get_frame(self):
        import numpy as np

        self._update_position()
        frame = self._background()
        self._composite(frame, self._sprite_frame())
        self.frame_index += 1
        return True, frame


def create_source(kind: str, **kwargs) -> FrameSource:
    """Kleine Factory, um die Quelle über einen String/Config-Wert zu wählen."""
    if kind == "video_file":
        return VideoFileSource(kwargs["path"], loop=kwargs.get("loop", True))
    if kind == "webcam":
        return WebcamSource(kwargs.get("index", 0))
    if kind == "synthetic":
        allowed = {"width", "height", "speed", "sprite_path"}
        return SyntheticMotionSource(**{k: v for k, v in kwargs.items() if k in allowed})
    raise ValueError(f"Unbekannte Quelle: {kind}")
