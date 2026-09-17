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

    def test_overlay_marks_motion_pixels_white(self):
        detector = MotionDetector(min_area=100, max_area=500)
        detector.bg_subtractor = StaticMask()
        frame = np.zeros((20, 20), dtype=np.uint8)

        detector.detect(frame)
        detector.overlay_motion_mask(frame)

        self.assertTrue(np.all(frame == 255))

    def test_overlay_supports_grayscale_frames(self):
        detector = MotionDetector(min_area=1, max_area=500)
        detector.bg_subtractor = StaticMask()
        frame = np.zeros((20, 20), dtype=np.uint8)

        detector.detect(frame)
        result = detector.overlay_motion_mask(frame)

        self.assertEqual(result.shape, (20, 20))
        self.assertTrue(np.all(result == 255))


if __name__ == "__main__":
    unittest.main()