from abc import ABC, abstractmethod
import numpy as np
from scipy import linalg
import logging

from data_imp.reg.reg_meta import Regressor
from data_imp.utils.losses import SquaredErrorLoss


class Kernel(ABC):
    @abstractmethod
    def k(self, x1, x2):
        pass


class GaussianKernel(Kernel):
    def __init__(self, c=None):
        if c is None:
            self.c = 1 / 2
        else:
            self.c = c

    def k(self, x1, x):
        eucld_dist = np.linalg.norm(x1 - x, axis=-1) / len(x1)
        return np.exp(- (self.c * self.c) * eucld_dist * eucld_dist)


class LocallyWeightedLinearRegression(Regressor):
    """
    Implementation of a locally weighted linear regression as described by Cleveland, 1979.
    """
    def __init__(self, kernel=None, loss=None, c=None, scaling=True, random_state=0):
        super().__init__(scaling, random_state)
        if kernel is not None:
            self.kernel = kernel
        else:
            self.kernel = GaussianKernel(c=c)

        if loss is not None:
            self.loss = loss
        else:
            self.loss = SquaredErrorLoss()

    def fit(self, X, y):
        X = self.fit_scaler(X)

        self.X = X
        self.y = y
        self._is_fitted = True
        return self

    def predict_single_instance(self, x):
        np.random.seed(self.random_state)

        x = self.scale_features(x)

        # Goal: Determine beta parameters by minimizing a sum of weighted losses
        d = self.X.shape[1]

        weights = self.kernel.k(x, self.X)

        b = np.zeros(d + 1)
        b[0] = np.sum(weights * self.y)
        for i in range(1, d + 1):
            b[i] = np.sum(weights * self.y * self.X[:, i - 1])

        A = np.zeros([d + 1, d + 1])
        A[0, 0] = np.sum(weights)
        for i in range(1, d + 1):
            A[0, i] = np.sum(weights * self.X[:, i - 1])

        for i in range(1, d + 1):
            A[i, 0] = np.sum(weights * self.X[:, i - 1])
            for j in range(1, d + 1):
                A[i, j] = np.sum(weights * self.X[:, j - 1] * self.X[:, i - 1])

        try:
            theta = linalg.solve(A, b)
        except np.linalg.LinAlgError:
            logging.warning("LinAlgError occurred, could not determine prediction (probably due to a singular matrix).")
            return 0

        assert len(theta) == d + 1

        if np.isnan(theta).any():
            logging.warning("Warning: Solving theta for A and b returned NaN values. 0 is returned as prediction.")
            return 0

        result = theta[0] + np.dot(theta[1:], x)

        return result
