import unittest
import numpy as np
from sklearn.metrics import zero_one_loss
import logging

from data_imp.clf.owsvm import OWSVM
from data_imp.utils.eval_utils import perform_complex_cv
from data_imp.utils.io_utils import read_dataset_from_openml


class OWSVMTest(unittest.TestCase):
    def setUp(self):
        self.clf_fn = OWSVM
        self.static_params = None

        self.param_dict = {"C": [1 / 100, 0.1, 0.5, 1, 5, 10]}

        logging.basicConfig(level=logging.INFO)

    def test_clf_random_data(self):
        np.random.seed(42)
        X = np.random.rand(10, 10)
        y = np.random.choice([-1, +1], 10)

        clf = self.clf_fn()
        clf.fit(X, y)

        x_test = np.random.rand(10)

        logging.info(clf.predict([x_test]))

    def test_clf_wbcd(self):
        # breast cancer wisconsin
        X, y = read_dataset_from_openml(1510)

        eval_result = perform_complex_cv(self.clf_fn, self.param_dict, X, y, zero_one_loss, num_outer_folds=5,
                                         static_param_dict=self.static_params)

        logging.info("Final CV score: {}".format(eval_result))

    def test_clf_banknote(self):
        # Banknote
        X, y = read_dataset_from_openml(1462)

        eval_result = perform_complex_cv(self.clf_fn, self.param_dict, X, y, zero_one_loss, num_outer_folds=5,
                                         static_param_dict=self.static_params)

        logging.info("Final CV score: {}".format(eval_result))

    def test_clf_parkinsons(self):
        # Parkinson's Disease Detection
        X, y = read_dataset_from_openml(1488)

        eval_result = perform_complex_cv(self.clf_fn, self.param_dict, X, y, zero_one_loss, num_outer_folds=5,
                                         static_param_dict=self.static_params)

        logging.info("Final CV score: {}".format(eval_result))
