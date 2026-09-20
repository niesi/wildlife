import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import unittest

import numpy as np

from motion_detector import MotionDetector


class StaticMask:
    def apply(self, frame, **kwargs):
        return np.full(frame.shape[:2], 255, dtype=np.uint8)


class MotionDetectorAreaTest(unittest.TestCase):
    def test_stores_minimum_and_maximum_area(self):
        detector = MotionDetector(min_area=100, max_area=500)

        self.assertEqual(detector.min_area, 100)
        self.assertEqual(detector.max_area, 500)

    def test_rejects_full_frame_foreground_region(self):
        detector = MotionDetector(min_area=100, max_area=500, downscale_width=100)
        detector.bg_subtractor = StaticMask()

        boxes = detector.detect(np.zeros((100, 100, 3), dtype=np.uint8))

        self.assertEqual(boxes, [])

    def test_overlay_marks_motion_pixels_red_without_changing_input(self):
        detector = MotionDetector(min_area=100, max_area=500)
        detector.bg_subtractor = StaticMask()
        frame = np.zeros((20, 20), dtype=np.uint8)

        detector.detect(frame)
        result = detector.overlay_motion_mask(frame)

        self.assertTrue(np.all(result == (0, 0, 255)))
        self.assertTrue(np.all(frame == 0))
        self.assertFalse(np.shares_memory(frame, result))

    def test_overlay_preserves_unmasked_pixels_and_blends(self):
        detector = MotionDetector()
        detector.last_mask = np.zeros((20, 20), dtype=np.uint8)
        detector.last_mask[0, 0] = 255
        frame = np.full((20, 20, 3), 100, dtype=np.uint8)
        frame[1, 1] = (10, 20, 30)
        original = frame.copy()

        result = detector.overlay_motion_mask(frame, alpha=0.5)

        np.testing.assert_array_equal(result[0, 0], (50, 50, 178))
        np.testing.assert_array_equal(result[1:], original[1:])
        np.testing.assert_array_equal(frame, original)

    def test_overlay_without_mask_returns_independent_color_image(self):
        detector = MotionDetector()
        frame = np.full((20, 20), 100, dtype=np.uint8)
        result = detector.overlay_motion_mask(frame)

        self.assertEqual(result.shape, (20, 20, 3))
        self.assertTrue(np.all(result == 100))
        self.assertFalse(np.shares_memory(frame, result))


if __name__ == "__main__":
    unittest.main()