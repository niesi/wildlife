"""Standalone FPS checks: run directly with Python (no camera needed)."""
import unittest
from unittest.mock import patch

import cv2

from frame_source import VideoFileSource
from pipeline import playback_delay_ms


class PlaybackTimingTest(unittest.TestCase):
    def test_video_reader_uses_reported_fps(self):
        with patch("frame_source.cv2.VideoCapture") as capture:
            capture.return_value.isOpened.return_value = True
            capture.return_value.get.return_value = 25.0
            source = VideoFileSource("known-fps.mp4", loop=False)
            self.assertEqual(source.fps, 25.0)
            capture.return_value.get.assert_called_once_with(cv2.CAP_PROP_FPS)
            source.release()
            capture.return_value.release.assert_called_once()

    def test_invalid_metadata_uses_fallback(self):
        for fps in (0.0, -1.0, float("nan"), float("inf")):
            with self.subTest(fps=fps), patch("frame_source.cv2.VideoCapture") as capture:
                capture.return_value.isOpened.return_value = True
                capture.return_value.get.return_value = fps
                source = VideoFileSource("unknown-fps.mp4")
                self.assertEqual(source.fps, 30.0)
                source.release()

    def test_delay_accounts_for_processing(self):
        # Exactly representable times avoid rounding at a millisecond boundary.
        with patch("pipeline.time.perf_counter", return_value=10.015625):
            self.assertEqual(playback_delay_ms(25.0, 10.0), 25)

    def test_full_frame_interval(self):
        with patch("pipeline.time.perf_counter", return_value=10.0):
            self.assertEqual(playback_delay_ms(25.0, 10.0), 40)
            self.assertEqual(playback_delay_ms(29.97, 10.0), 34)

    def test_slow_processing_still_pumps_gui(self):
        with patch("pipeline.time.perf_counter", return_value=10.125):
            self.assertEqual(playback_delay_ms(25.0, 10.0), 1)

    def test_non_video_sources_keep_existing_delay(self):
        self.assertEqual(playback_delay_ms(None, 10.0), 1)


if __name__ == "__main__":
    unittest.main()
