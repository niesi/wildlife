import unittest

from frame_source import SyntheticMotionSource


class SyntheticMotionSourceTest(unittest.TestCase):
    def test_get_frame_supports_random_xy_motion(self):
        source = SyntheticMotionSource(width=100, height=100, speed=6)
        ok, frame = source.get_frame()

        self.assertTrue(ok)
        self.assertEqual(frame.shape, (100, 100, 3))
        self.assertTrue(hasattr(source, "x"))
        self.assertTrue(hasattr(source, "y"))
        self.assertLessEqual(source.x, source.width + source.sprite_w)
        self.assertGreaterEqual(source.y, 0)
        self.assertLessEqual(source.y, source.height)


if __name__ == "__main__":
    unittest.main()
