from data_imp.reg.interval_reg import IntervalBasedLinearRegression
from test.data_imp.reg.reg_test_meta import RegressionModelTest


class IntervalBasedLinearRegressionTest(RegressionModelTest):
    def setUp(self):
        super().setUp()

        self.regressor_fn = IntervalBasedLinearRegression
        self.param_dict = {"c": [1 / 100, 1 / 10, 1 / 5, 1 / 2, 1, 2]}