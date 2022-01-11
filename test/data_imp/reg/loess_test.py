from data_imp.reg.loess import LocallyWeightedLinearRegression
from test.data_imp.reg.reg_test_meta import RegressionModelTest


class LocallyWeightedLinearRegressionTest(RegressionModelTest):
    def setUp(self):
        super().setUp()

        self.regressor_fn = LocallyWeightedLinearRegression
        self.param_dict = {"c": [1 / 100, 1 / 10, 1 / 5, 1 / 2, 1, 2]}