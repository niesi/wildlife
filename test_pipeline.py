import unittest
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from motion_detector import MotionDetector
from pipeline import run


class PipelineDisplayTest(unittest.TestCase):
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
                for call, color in zip(display.call_args_list, ((0, 220, 0), (0, 200, 255))):
                    image = call.args[1]
                    self.assertEqual(image.shape, (100, 100, 3))
                    np.testing.assert_array_equal(image[30, 30], color)
                    np.testing.assert_array_equal(image[99, 99], (80, 80, 80))
                    if show_mask:
                        np.testing.assert_array_equal(image[85, 85], (0, 0, 255))
                    self.assertFalse(np.shares_memory(image, frame))
                assert_gray(frame)
                source.release.assert_called_once()
                cleanup.assert_called_once()


if __name__ == "__main__":
    unittest.main()
