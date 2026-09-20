import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import unittest
from types import SimpleNamespace

import numpy as np

from target_detector import CatTargetDetector, Target, box_center, expand_box


FRAME = np.full((240, 320), 80, dtype=np.uint8)
BOX = (140, 100, 40, 40)


class FakeMotion:
    """Motion detector stand-in returning a fixed list of boxes."""

    def __init__(self, boxes=(BOX,)):
        self.boxes = list(boxes)
        self.calls = 0

    def detect(self, frame):
        self.calls += 1
        return list(self.boxes)


class FakeClassifier:
    """Classifier stand-in: fixed label or a scripted label sequence."""

    def __init__(self, labels=None, label="cat", confidence=0.9):
        self.labels = list(labels) if labels is not None else None
        self.label = label
        self.confidence = confidence
        self.calls = 0
        self.crops = []

    def classify(self, crop):
        self.calls += 1
        self.crops.append(crop)
        label = self.labels.pop(0) if self.labels else self.label
        return SimpleNamespace(label=label, confidence=self.confidence, is_animal=label == "cat")


class FakeTracker:
    """Tracker stand-in that follows a fixed box until it is told to fail."""

    def __init__(self, ok=True, box=BOX, reclassify=False):
        self.active = False
        self.label = None
        self.frames_since_start = 0
        self.ok = ok
        self.box = box
        self.reclassify = reclassify
        self.starts = 0
        self.stops = 0

    def start(self, frame, box, label):
        self.starts += 1
        self.active = True
        self.box = box
        self.label = label
        self.frames_since_start = 0

    def update(self, frame):
        if not self.active:
            return False, None
        self.frames_since_start += 1
        if self.ok and self.box is not None:
            return True, tuple(self.box)
        self.active = False
        return False, None

    def needs_reclassification(self):
        return self.reclassify

    def stop(self):
        self.stops += 1
        self.active = False


def detector_with(motion=None, classifier=None, tracker=None, **kwargs):
    motion = motion if motion is not None else FakeMotion()
    classifier = classifier if classifier is not None else FakeClassifier()
    tracker = tracker if tracker is not None else FakeTracker()
    return CatTargetDetector(motion, classifier, tracker, **kwargs), motion, classifier, tracker


class DetectionFlowTest(unittest.TestCase):
    def test_no_motion_returns_no_target(self):
        detector, _, classifier, _ = detector_with(motion=FakeMotion(boxes=[]))

        self.assertIsNone(detector.update(FRAME, now=0.0))
        self.assertEqual(classifier.calls, 0)

    def test_foreign_label_is_ignored(self):
        detector, _, _, tracker = detector_with(classifier=FakeClassifier(label="dog"))

        self.assertIsNone(detector.update(FRAME, now=0.0))
        self.assertFalse(tracker.active)

    def test_first_frame_yields_an_unconfirmed_target(self):
        detector, _, _, _ = detector_with()

        target = detector.update(FRAME, now=1.0)

        self.assertIsInstance(target, Target)
        self.assertEqual(target.label, "cat")
        self.assertEqual(target.confidence, 0.9)
        self.assertEqual(target.frames_confirmed, 1)
        self.assertEqual(target.box, BOX)
        self.assertEqual(target.center, (160.0, 120.0))
        self.assertEqual(target.first_seen_at, 1.0)
        self.assertEqual(target.last_seen_at, 1.0)

    def test_confirmation_counter_grows_with_consecutive_frames(self):
        detector, _, _, _ = detector_with()

        targets = [detector.update(FRAME, now=float(index)) for index in range(3)]

        # frame one starts the tracker, so later frames are served by it
        self.assertEqual([target.frames_confirmed for target in targets], [1, 2, 3])

    def test_counter_restarts_after_a_gap(self):
        classifier = FakeClassifier(labels=["cat", "dog", "cat"])
        detector, _, _, _ = detector_with(classifier=classifier)

        detector.update(FRAME, now=0.0)
        detector.tracker.stop()
        self.assertIsNone(detector.update(FRAME, now=1.0))
        detector.tracker.stop()
        target = detector.update(FRAME, now=2.0)

        self.assertEqual(target.frames_confirmed, 1)

    def test_tracker_starts_once_while_the_candidate_is_visible(self):
        detector, _, _, tracker = detector_with()

        detector.update(FRAME, now=0.0)
        detector.update(FRAME, now=1.0)

        self.assertEqual(tracker.starts, 1)
        self.assertTrue(tracker.active)
        self.assertEqual(tracker.label, "cat")

    def test_box_outside_the_frame_is_skipped(self):
        motion = FakeMotion(boxes=[(400, 300, 10, 10), BOX])
        detector, _, _, _ = detector_with(motion=motion)

        target = detector.update(FRAME, now=0.0)

        self.assertEqual(target.box, BOX)


class TrackingTest(unittest.TestCase):
    def test_tracked_target_keeps_its_confirmation_history(self):
        tracker = FakeTracker(reclassify=True)
        detector, _, classifier, _ = detector_with(tracker=tracker)

        detector.update(FRAME, now=0.0)
        target = detector.update(FRAME, now=0.5)

        self.assertEqual(target.frames_confirmed, 2)
        self.assertEqual(classifier.calls, 2)
        self.assertEqual(tracker.frames_since_start, 0)
        self.assertEqual(tracker.label, "cat")

    def test_tracking_is_dropped_when_reclassification_rejects(self):
        classifier = FakeClassifier(labels=["cat", "dog"])
        tracker = FakeTracker(reclassify=True)
        detector, _, _, _ = detector_with(classifier=classifier, tracker=tracker)

        detector.update(FRAME, now=0.0)

        self.assertIsNone(detector.update(FRAME, now=0.5))
        self.assertFalse(tracker.active)
        self.assertEqual(detector.candidate_frames, 0)

    def test_lost_tracker_returns_none_and_resets(self):
        tracker = FakeTracker(ok=False)
        detector, _, _, _ = detector_with(tracker=tracker)

        detector.update(FRAME, now=0.0)

        self.assertIsNone(detector.update(FRAME, now=0.5))
        self.assertFalse(tracker.active)
        self.assertEqual(detector.candidate_frames, 0)

    def test_empty_frame_resets_the_candidate(self):
        detector, _, _, tracker = detector_with()
        detector.update(FRAME, now=0.0)

        empty = np.empty((0, 0), dtype=np.uint8)

        self.assertIsNone(detector.update(empty, now=1.0))
        self.assertEqual(detector.candidate_frames, 0)
        self.assertGreaterEqual(tracker.stops, 1)

    def test_velocity_is_estimated_from_the_sample_history(self):
        motion = FakeMotion()
        detector, _, _, _ = detector_with(motion=motion)

        motion.boxes = [(140, 100, 40, 40)]
        detector.update(FRAME, now=0.0)
        detector.tracker.stop()
        motion.boxes = [(170, 100, 40, 40)]
        detector.update(FRAME, now=0.5)
        detector.tracker.stop()
        motion.boxes = [(200, 100, 40, 40)]
        target = detector.update(FRAME, now=1.0)

        self.assertAlmostEqual(target.velocity_px_s[0], 60.0)
        self.assertAlmostEqual(target.velocity_px_s[1], 0.0)

    def test_seen_for_reports_the_observation_window(self):
        detector, _, _, _ = detector_with()

        detector.update(FRAME, now=10.0)
        target = detector.update(FRAME, now=12.5)

        self.assertAlmostEqual(target.seen_for_s, 2.5)


class GeometryTest(unittest.TestCase):
    def test_box_center(self):
        self.assertEqual(box_center((10, 20, 4, 6)), (12.0, 23.0))

    def test_expand_box_is_clipped_to_the_frame(self):
        self.assertEqual(expand_box((0, 0, 10, 10), (100, 100), margin=20), (0, 0, 30, 30))
        self.assertEqual(expand_box((95, 95, 10, 10), (100, 100), margin=20), (75, 75, 100, 100))

    def test_crop_is_expanded_by_the_margin(self):
        detector, _, classifier, _ = detector_with(crop_margin=10)

        detector.update(FRAME, now=0.0)

        self.assertEqual(classifier.crops[0].shape, (60, 60))

    def test_negative_margin_is_clamped_to_zero(self):
        detector, _, classifier, _ = detector_with(crop_margin=-5)

        detector.update(FRAME, now=0.0)

        self.assertEqual(classifier.crops[0].shape, (40, 40))

    def test_frames_are_expected_to_be_grayscale(self):
        detector, _, classifier, _ = detector_with()

        detector.update(FRAME, now=0.0)

        self.assertEqual(classifier.crops[0].ndim, 2)


if __name__ == "__main__":
    unittest.main()

