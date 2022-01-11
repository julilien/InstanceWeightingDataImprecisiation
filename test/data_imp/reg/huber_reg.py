from data_imp.reg.huber import HuberWrapper
from test.data_imp.reg.reg_test_meta import RegressionModelTest


class HuberRegressionTest(RegressionModelTest):
    def setUp(self):
        super().setUp()

        self.regressor_fn = HuberWrapper
        self.param_dict = {"epsilon": [1, 1.2, 1.35, 1.5, 2], "alpha": [0.0001, 0.001, 0.01, 0.1, 1]}