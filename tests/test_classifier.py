import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import unittest

import numpy as np

from classifier import MockClassifier


class MockClassifierAreaTest(unittest.TestCase):
    def test_accepts_area_above_minimum(self):
        classifier = MockClassifier(min_area_for_animal=100)

        result = classifier.classify(np.zeros((20, 10, 3), dtype=np.uint8))

        self.assertTrue(result.is_animal)

    def test_accepts_large_crop_when_above_minimum(self):
        classifier = MockClassifier(min_area_for_animal=100)

        result = classifier.classify(np.zeros((25, 20, 3), dtype=np.uint8))

        self.assertTrue(result.is_animal)
        self.assertEqual(result.label, "animal")


if __name__ == "__main__":
    unittest.main()