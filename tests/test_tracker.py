import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import unittest

import numpy as np

from tracker import AnimalTracker


class AnimalTrackerTest(unittest.TestCase):
    def test_update_ignores_empty_frame_and_returns_false(self):
        tracker = AnimalTracker()
        frame = np.zeros((20, 20, 3), dtype=np.uint8)
        tracker.start(frame, (0, 0, 10, 10), "animal")

        empty_frame = np.empty((0, 0, 3), dtype=np.uint8)
        ok, box = tracker.update(empty_frame)

        self.assertFalse(ok)
        self.assertIsNone(box)
        self.assertFalse(tracker.active)


if __name__ == "__main__":
    unittest.main()
