import ray
import unittest
import logging
import numpy as np
from sklearn.metrics import mean_squared_error

from data_imp.utils.eval_utils import perform_complex_cv
from data_imp.utils.io_utils import read_dataset_from_openml

class RegressionModelTest(unittest.TestCase):
    def setUp(self):
        self.regressor_fn = None
        self.param_dict = None

        logging.basicConfig(level=logging.DEBUG)

    def test_reg_with_random_data(self):
        np.random.seed(42)
        X = np.random.rand(10, 10)
        y = np.random.choice(3, 10)

        reg = self.regressor_fn()
        reg = reg.fit(X, y)

        x_test = np.random.rand(10)

        logging.info(reg.predict(x_test))

    def execute_experiment(self, dataset_id, target_selector=-1):
        X, y = read_dataset_from_openml(dataset_id, classification=False, target_selector=target_selector)

        eval_result = perform_complex_cv(self.regressor_fn, self.param_dict, X, y, mean_squared_error)
        logging.info("Final CV score: {}".format(eval_result))

    def test_reg_wine_quality_red(self):
        self.execute_experiment(40691)

    def test_reg_wine_quality_white(self):
        self.execute_experiment(40498)

    def test_reg_community_crimes(self):
        self.execute_experiment(41969)

    def test_reg_parkinsons(self):
        # Select attribute total_UPDRS
        self.execute_experiment(4531, target_selector=5)

    def test_reg_wpbc(self):
        self.execute_experiment(191)

    def test_reg_stat_lib(self):
        self.execute_experiment(210)