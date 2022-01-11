import unittest
import numpy as np

from data_imp.exp.exp_utils import fuzzify_labels, get_model_parameters


class ExpUtilsTest(unittest.TestCase):
    def test_fuzzify_labels(self):
        y = np.random.choice([-1, 1], 10)
        y_flipped = fuzzify_labels(y, 1.0, 0)
        self.assertTrue(np.array_equal(y * -1, y_flipped))

        y_same = fuzzify_labels(y, 0.0, 0)
        self.assertTrue(np.array_equal(y, y_same))
