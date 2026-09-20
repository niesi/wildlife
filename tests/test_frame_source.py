import unittest
from unittest.mock import patch

import cv2
import numpy as np

from frame_source import SyntheticMotionSource, VideoFileSource, WebcamSource


class SyntheticMotionSourceTest(unittest.TestCase):
    def test_get_frame_supports_random_xy_motion(self):
        source = SyntheticMotionSource(width=100, height=100, speed=6)
        ok, frame = source.get_frame()

        self.assertTrue(ok)
        self.assertEqual(frame.shape, (100, 100))
        self.assertTrue(hasattr(source, "x"))
        self.assertTrue(hasattr(source, "y"))
        self.assertLessEqual(source.x, source.width + source.sprite_w)
        self.assertGreaterEqual(source.y, 0)
        self.assertLessEqual(source.y, source.height)


class CaptureSourceTest(unittest.TestCase):
    def test_video_and_webcam_return_grayscale(self):
        color = np.full((20, 30, 3), (30, 100, 200), dtype=np.uint8)
        expected = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
        for source_type, value in ((VideoFileSource, "clip.mp4"), (WebcamSource, 0)):
            with self.subTest(source=source_type), patch("frame_source.cv2.VideoCapture") as capture:
                capture.return_value.isOpened.return_value = True
                capture.return_value.get.return_value = 25.0
                capture.return_value.read.return_value = (True, color.copy())
                source = source_type(value)
                try:
                    ok, frame = source.get_frame()
                    self.assertTrue(ok)
                    np.testing.assert_array_equal(frame, expected)
                finally:
                    source.release()


if __name__ == "__main__":
    unittest.main()
