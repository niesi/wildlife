"""
Turns motion regions into a target the turret can aim at.

This module glues the existing perception pieces together (motion detector,
classifier, tracker) and counts how many consecutive frames a motion region
was classified as the target class. The resulting :class:`Target` carries the
``frames_confirmed`` counter the safety gate checks - deciding whether to
spray stays outside, in ``safety.py`` and ``turret_controller.py``.

Frames are expected to be grayscale, exactly like in the wildlife pipeline;
only the display layer converts to colour.
"""

from dataclasses import dataclass


TARGET_LABEL = "cat"
NEGATIVE_LABEL = "not_cat"


@dataclass(frozen=True)
class Target:
    """A cat candidate: position in the frame plus confirmation state."""

    box: tuple                  # x, y, w, h
    center: tuple               # cx, cy
    label: str
    confidence: float
    frames_confirmed: int       # consecutive frames with the target class
    first_seen_at: float
    last_seen_at: float
    velocity_px_s: tuple = (0.0, 0.0)

    @property
    def seen_for_s(self) -> float:
        """How long this candidate has been visible."""
        return self.last_seen_at - self.first_seen_at


def box_center(box) -> tuple:
    x, y, w, h = box
    return (x + w / 2.0, y + h / 2.0)


def expand_box(box, frame_shape, margin=20):
    """Grow a box for cropping, clipped to the frame (mirrors pipeline.expand_box)."""
    x, y, w, h = box
    frame_h, frame_w = frame_shape[:2]
    x0 = max(0, int(x) - margin)
    y0 = max(0, int(y) - margin)
    x1 = min(frame_w, int(x + w) + margin)
    y1 = min(frame_h, int(y + h) + margin)
    return x0, y0, x1, y1


class CatTargetDetector:
    """
    Motion -> classification -> tracking for one cat at a time.

    The three collaborators are injected (``detector`` for motion,
    ``classifier`` for the label, ``tracker`` for following a confirmed
    candidate), so the whole flow can be tested with plain stand-ins instead
    of camera frames.

    ``frames_confirmed`` counts consecutive frames in which a motion region
    was classified as one of ``target_labels``. Only the controller compares
    that counter with ``SafetyLimits.confirm_frames``.
    """

    def __init__(self, detector, classifier, tracker, target_labels=(TARGET_LABEL,),
                 crop_margin=20, reclassify_interval_frames=60, velocity_window=5):
        self.detector = detector
        self.classifier = classifier
        self.tracker = tracker
        self.target_labels = tuple(target_labels)
        self.crop_margin = max(0, int(crop_margin))
        self.reclassify_interval_frames = reclassify_interval_frames
        self.velocity_window = max(2, int(velocity_window))
        self._candidate_frames = 0
        self._first_seen_at = None
        self._label = None
        self._confidence = 0.0
        self._samples = []

    def reset(self):
        """Forget the candidate and stop the tracker."""
        self._candidate_frames = 0
        self._first_seen_at = None
        self._label = None
        self._confidence = 0.0
        self._samples = []
        if self.tracker is not None:
            self.tracker.stop()

    @property
    def candidate_frames(self) -> int:
        """Consecutive frames with a matching candidate (0 = none)."""
        return self._candidate_frames

    def update(self, frame, now):
        """
        Return the current :class:`Target` (possibly not confirmed yet) or
        ``None`` when nothing of the target class is visible.
        """
        if frame is None or getattr(frame, "size", None) == 0:
            self.reset()
            return None

        if self.tracker is not None and self.tracker.active:
            target = self._update_tracked(frame, now)
            if target is not None:
                return target
            return None

        return self._search(frame, now)

    # --- internals ---------------------------------------------------------
    def _search(self, frame, now):
        for box in self.detector.detect(frame):
            if box is None or len(box) != 4:
                continue
            crop = self._crop(frame, box)
            if crop is None:
                continue
            result = self.classifier.classify(crop)
            if result.label not in self.target_labels:
                continue
            self._count_candidate(result, now)
            if self.tracker is not None and not self.tracker.active:
                self.tracker.start(frame, box, result.label)
            return self._target(box, now)

        self._forget_candidate()
        return None

    def _update_tracked(self, frame, now):
        ok, box = self.tracker.update(frame)
        if not ok or box is None:
            self.reset()
            return None

        if self.tracker.needs_reclassification():
            crop = self._crop(frame, box)
            if crop is not None:
                result = self.classifier.classify(crop)
                if result.label not in self.target_labels:
                    self.reset()
                    return None
                self._label = result.label
                self._confidence = result.confidence
                self.tracker.label = result.label
                self.tracker.frames_since_start = 0

        self._candidate_frames += 1
        return self._target(box, now)

    def _count_candidate(self, result, now):
        self._candidate_frames += 1
        self._label = result.label
        self._confidence = result.confidence
        if self._first_seen_at is None:
            self._first_seen_at = now

    def _forget_candidate(self):
        self._candidate_frames = 0
        self._first_seen_at = None
        self._samples = []

    def _crop(self, frame, box):
        x0, y0, x1, y1 = expand_box(box, frame.shape, self.crop_margin)
        if x1 <= x0 or y1 <= y0:
            return None
        crop = frame[y0:y1, x0:x1]
        if getattr(crop, "size", None) == 0:
            return None
        return crop

    def _target(self, box, now):
        center = box_center(box)
        self._samples.append((now, center))
        del self._samples[:-self.velocity_window]
        first_seen = self._first_seen_at if self._first_seen_at is not None else now
        return Target(
            box=tuple(int(value) for value in box),
            center=center,
            label=self._label,
            confidence=float(self._confidence),
            frames_confirmed=self._candidate_frames,
            first_seen_at=first_seen,
            last_seen_at=now,
            velocity_px_s=self._velocity(),
        )

    def _velocity(self):
        if len(self._samples) < 2:
            return (0.0, 0.0)
        (t0, (x0, y0)), (t1, (x1, y1)) = self._samples[0], self._samples[-1]
        dt = t1 - t0
        if dt <= 0:
            return (0.0, 0.0)
        return ((x1 - x0) / dt, (y1 - y0) / dt)