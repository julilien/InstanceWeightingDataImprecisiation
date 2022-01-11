from sklearn.linear_model import HuberRegressor

from data_imp.reg.reg_meta import Regressor

import numpy as np


class HuberWrapper(Regressor):
    """
    Wrapper of a Huber-loss based regressor.
    """
    def __init__(self, epsilon=1.35, alpha=0.1, scaling=True, random_state=0):
        super().__init__(scaling, random_state)
        self.model = HuberRegressor(epsilon=epsilon, alpha=alpha, max_iter=1000)

    def fit(self, X, y):
        np.random.seed(self.random_state)
        X = self.fit_scaler(X)

        self.model.fit(X, y)
        return self

    def predict_single_instance(self, x):
        x = self.scale_features(x)

        if x.ndim == 1:
            x = x.reshape(1, -1)

        return self.model.predict(x)

