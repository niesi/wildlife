import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import unittest
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from motion_detector import MotionDetector
from pipeline import run, draw_stage_status, ACCEPT_COLOR, TRACKER_COLOR


class StageStatusTest(unittest.TestCase):
    def test_three_labeled_rows_render_below_unchanged_video(self):
        import cv2

        frame = np.full((100, 160, 3), 80, dtype=np.uint8)
        original = frame.copy()
        with patch("pipeline.cv2.putText", wraps=cv2.putText) as text:
            output = draw_stage_status(frame, "2 regions", "tracking tier",
                                       "tier 0.90 - fresh", ACCEPT_COLOR)
        labels = [call.args[1] for call in text.call_args_list]
        self.assertEqual(labels, ["MOTION: 2 regions", "TRACKER: tracking tier",
                                 "CLASSIFICATION (MOCK): tier 0.90 - fresh"])
        self.assertEqual(output.shape[0], 196)
        for index, call in enumerate(text.call_args_list):
            self.assertEqual(call.args[2], (12, 124 + index * 30))
            row = output[100 + index * 30:130 + index * 30]
            self.assertTrue(np.any(row[:, :, 0] != row[:, :, 1]))
        np.testing.assert_array_equal(frame, original)
        np.testing.assert_array_equal(output[:100, :160], original)



class PipelineDisplayTest(unittest.TestCase):
    def test_run_renders_exact_stage_results_end_to_end(self):
        import cv2

        with ExitStack() as stack:
            frame = np.full((120, 160), 80, dtype=np.uint8)
            source = Mock()
            source.get_frame.return_value = (True, frame)
            motion = Mock()
            motion.detect.return_value = [(40, 60, 20, 20)]
            classifier = Mock()
            classifier.classify.side_effect = [
                SimpleNamespace(label="tier", confidence=0.9, is_animal=True),
                SimpleNamespace(label="kein_tier", confidence=0.2, is_animal=False),
            ]
            tracker = Mock(active=False, label="tier")
            tracker.start.side_effect = lambda *args: setattr(tracker, "active", True)
            tracker.stop.side_effect = lambda: setattr(tracker, "active", False)
            tracker.update.return_value = (True, (40, 60, 20, 20))
            tracker.needs_reclassification.return_value = True
            for name, instance in (("create_source", source), ("MotionDetector", motion),
                                   ("MockClassifier", classifier), ("AnimalTracker", tracker)):
                stack.enter_context(patch("pipeline." + name, return_value=instance))
            displayed = []
            stack.enter_context(patch("pipeline.cv2.imshow",
                                      side_effect=lambda title, image: displayed.append(image.copy())))
            stack.enter_context(patch("pipeline.cv2.waitKey", side_effect=[-1, ord("q")]))
            stack.enter_context(patch("pipeline.cv2.destroyAllWindows"))
            real_put_text = cv2.putText
            text = stack.enter_context(patch("pipeline.cv2.putText", wraps=real_put_text))
            run(SimpleNamespace(source="synthetic", path=None, index=0, sprite=None,
                                min_area=100, max_area=50000, min_animal_area=100,
                                reclassify_every=1, show_motion_mask=False))

            blue, yellow, green, red = (255, 160, 0), (0, 200, 255), (0, 220, 0), (0, 0, 220)
            expected_rows = [
                [("MOTION: 1 regions", blue), ("TRACKER: started tier", yellow),
                 ("CLASSIFICATION (MOCK): tier 0.90 | accepted | fresh (1 crops, last result)", green)],
                [("MOTION: skipped while tracking", blue),
                 ("TRACKER: stopped: classification rejected", yellow),
                 ("CLASSIFICATION (MOCK): kein_tier 0.20 | rejected | fresh (1 crops, last result)", red)],
            ]
            actual_text = [(call.args[1], call.args[5]) for call in text.call_args_list]
            self.assertEqual(actual_text, [
                ("MOTION #1", blue), ("CLASSIFY: tier 0.90", green), ("TRACKER: tier", yellow),
                *expected_rows[0], ("TRACKER: tier", yellow), ("CLASSIFY: kein_tier 0.20", red),
                *expected_rows[1],
            ])
            self.assertEqual(len(displayed), 2)
            for image, rows, classification_color in zip(displayed, expected_rows, (green, red)):
                self.assertEqual(image.shape[0], 216)
                self.assertEqual(image.shape[2], 3)
                expected_panel = np.full((96, image.shape[1], 3), 20, dtype=np.uint8)
                for index, (label, color) in enumerate(rows):
                    real_put_text(expected_panel, label, (12, 24 + index * 30),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)
                np.testing.assert_array_equal(image[120:], expected_panel)
                np.testing.assert_array_equal(image[60, 40], yellow)
                np.testing.assert_array_equal(image[80, 20], classification_color)
                np.testing.assert_array_equal(image[119, 159], (80, 80, 80))
            self.assertTrue(np.all(frame == 80))
            self.assertFalse(tracker.active)
            self.assertEqual(classifier.classify.call_count, 2)
            motion.detect.assert_called_once()
            source.release.assert_called_once()

    def test_status_reports_idle_fresh_cached_and_lost_results(self):
        with ExitStack() as stack:
            frame = np.full((120, 160), 80, dtype=np.uint8)
            source = Mock()
            source.get_frame.return_value = (True, frame)
            motion = Mock()
            motion.detect.side_effect = [[], [(40, 40, 20, 20)]]
            classifier = Mock()
            classifier.classify.return_value = SimpleNamespace(
                label="tier", confidence=0.9, is_animal=True
            )
            tracker = Mock(active=False, label="tier")
            tracker.start.side_effect = lambda *args: setattr(tracker, "active", True)
            tracker.stop.side_effect = lambda: setattr(tracker, "active", False)
            tracker.update.side_effect = [(True, (40, 40, 20, 20)), (False, None)]
            tracker.needs_reclassification.return_value = False
            for name, instance in (("create_source", source), ("MotionDetector", motion),
                                   ("MockClassifier", classifier), ("AnimalTracker", tracker)):
                stack.enter_context(patch("pipeline." + name, return_value=instance))
            status = stack.enter_context(patch("pipeline.draw_stage_status", wraps=draw_stage_status))
            stack.enter_context(patch("pipeline.cv2.imshow"))
            stack.enter_context(patch("pipeline.cv2.waitKey", side_effect=[-1, -1, -1, ord("q")]))
            stack.enter_context(patch("pipeline.cv2.destroyAllWindows"))
            run(SimpleNamespace(source="synthetic", path=None, index=0, sprite=None,
                                min_area=100, max_area=50000, min_animal_area=100,
                                reclassify_every=60, show_motion_mask=False))
            rows = [call.args[1:] for call in status.call_args_list]
            self.assertEqual(rows[0][:3], ("0 regions", "idle", "not run yet"))
            self.assertEqual(rows[1][0:2], ("1 regions", "started tier"))
            self.assertIn("fresh", rows[1][2])
            self.assertIn("tier 0.90 | accepted", rows[1][2])
            self.assertEqual(rows[2][0:2], ("skipped while tracking", "tracking tier"))
            self.assertIn("1 frames ago", rows[2][2])
            self.assertEqual(rows[3][1], "lost")
            self.assertIn("2 frames ago", rows[3][2])
            classifier.classify.assert_called_once()
            self.assertEqual(motion.detect.call_count, 2)

    def test_processing_stays_gray_and_clean_with_colored_overlays(self):
        for show_mask in (False, True):
            with self.subTest(show_mask=show_mask), ExitStack() as stack:
                frame = np.full((100, 100), 80, dtype=np.uint8)
                source = Mock()
                source.get_frame.side_effect = [(True, frame), (True, frame)]
                motion = Mock()
                motion.detect.return_value = [(30, 30, 20, 20)]
                mask_renderer = MotionDetector()
                mask_renderer.last_mask = np.zeros(frame.shape, dtype=np.uint8)
                mask_renderer.last_mask[80:90, 80:90] = 255
                motion.overlay_motion_mask.side_effect = mask_renderer.overlay_motion_mask
                classifier = Mock()
                classifier.classify.return_value = SimpleNamespace(
                    is_animal=True, label="tier", confidence=0.9
                )
                tracker = Mock(active=False, label="tier")

                def assert_gray(image):
                    self.assertEqual(image.ndim, 2)
                    self.assertTrue(np.all(image == 80))

                def start(image, box, label):
                    assert_gray(image)
                    tracker.active = True

                def update(image):
                    assert_gray(image)
                    return True, (30, 30, 20, 20)

                tracker.start.side_effect = start
                tracker.update.side_effect = update
                tracker.needs_reclassification.return_value = True
                for name, instance in (("create_source", source), ("MotionDetector", motion),
                                       ("MockClassifier", classifier), ("AnimalTracker", tracker)):
                    stack.enter_context(patch("pipeline." + name, return_value=instance))
                display = stack.enter_context(patch("pipeline.cv2.imshow"))
                stack.enter_context(patch("pipeline.cv2.waitKey", side_effect=[-1, ord("q")]))
                cleanup = stack.enter_context(patch("pipeline.cv2.destroyAllWindows"))
                args = SimpleNamespace(source="synthetic", path=None, index=0, sprite=None,
                                       min_area=100, max_area=50000, min_animal_area=100,
                                       reclassify_every=1, show_motion_mask=show_mask)
                run(args)

                self.assertEqual(classifier.classify.call_count, 2)
                for call in classifier.classify.call_args_list + motion.detect.call_args_list:
                    assert_gray(call.args[0])
                tracker.start.assert_called_once()
                tracker.update.assert_called_once()
                self.assertEqual(display.call_count, 2)
                for call in display.call_args_list:
                    image = call.args[1]
                    self.assertEqual(image.shape[0], 196)
                    self.assertEqual(image.shape[2], 3)
                    np.testing.assert_array_equal(image[30, 30], TRACKER_COLOR)
                    np.testing.assert_array_equal(image[99, 99], (80, 80, 80))
                    if show_mask:
                        np.testing.assert_array_equal(image[85, 85], (0, 0, 255))
                    self.assertFalse(np.shares_memory(image, frame))
                assert_gray(frame)
                source.release.assert_called_once()
                cleanup.assert_called_once()


if __name__ == "__main__":
    unittest.main()
