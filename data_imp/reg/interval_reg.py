import numpy as np
from scipy import optimize
import logging

from sklearn.linear_model import LinearRegression

from data_imp.reg.reg_meta import Regressor
from data_imp.utils.losses import SquaredErrorLoss


class IntervalBasedLinearRegression(Regressor):
    """
    Implementation of a regressor using instance weighting through data imprecisiation.

    Assumption: Squared error loss is taken (reason: this eases the determination of y_n^* within the iterative optimization.
    """

    def __init__(self, kernel=None, c=None, num_iter=30, loss=None, scaling=True, random_state=0):
        super().__init__(scaling, random_state)
        if kernel is None:
            if c is None:
                self.c = 1 / 2
            else:
                self.c = c
            self.kernel_fn = lambda x, y: self.default_kernel(x, y, self.c)
        else:
            self.kernel_fn = kernel

        self.num_iter = num_iter
        if self.num_iter == -1:
            logging.warning("Warning: Setting the number of iterations to unlimited (i.e., -1) is not recommended since it could "
                  "drastically harm the performance due to very exhaustive runs.")
        if loss is None:
            self.loss = SquaredErrorLoss()
        else:
            self.loss = loss

    @staticmethod
    def default_kernel(x, x_prime, c):
        eucl_dists = np.linalg.norm(x- x_prime, axis=-1) / len(x)
        return np.exp(c * c * eucl_dists * eucl_dists) - 1

    @staticmethod
    def fit_std_lr(X, y):
        lr = LinearRegression()
        lr.fit(X, y)
        return lr.coef_, lr.intercept_

    def fit(self, X, y):
        X = self.fit_scaler(X)

        self.X = X
        self.y = y
        self._is_fitted = True
        return self

    def predict_single_instance(self, x):
        np.random.seed(self.random_state)

        x = self.scale_features(x)

        curr_y = np.zeros_like(self.y)

        # Determine updated curr_yns
        delta_ns = self.kernel_fn(x, self.X)
        fuzzy_y = np.vstack((self.y - delta_ns, self.y + delta_ns)).T

        beta_prime_coef, beta_prime_intercept = self.fit_std_lr(self.X, self.y)
        old_beta = np.zeros(len(beta_prime_coef) + 1)

        is_squared_err_loss = isinstance(self.loss, SquaredErrorLoss)

        curr_it = 0
        while np.linalg.norm(old_beta - np.hstack((beta_prime_intercept, beta_prime_coef))) > 0.001:
            if self.num_iter != -1 and curr_it > self.num_iter:
                logging.debug("The number of maximal iterations ({}) is reached. Aborting optimization...".format(self.num_iter))
                break

            y_hat = np.dot(self.X, beta_prime_coef) + beta_prime_intercept
            for i in range(self.X.shape[0]):
                try:
                    if np.abs(fuzzy_y[i, 0] - fuzzy_y[i, 1]) > 0.001:
                        if is_squared_err_loss:
                            # Get value closest to y_hat (but this is only valid for squared error loss!)
                            if fuzzy_y[i, 0] <= y_hat[i] <= fuzzy_y[i, 1]:
                                curr_y[i] = y_hat[i]
                            elif y_hat[i] < fuzzy_y[i, 0]:
                                curr_y[i] = fuzzy_y[i, 0]
                            else:
                                curr_y[i] = fuzzy_y[i, 1]
                        else:
                            min_fn = lambda y_true: self.loss.loss(y_true, y_hat[i])
                            curr_y[i] = optimize.fminbound(min_fn, fuzzy_y[i, 0], fuzzy_y[i, 1])
                except UnboundLocalError:
                    logging.warning(
                        "Could not get an optimized y_n value for the given fuzzy set "
                        "[{}, {}]. Keeping the current value.".format(fuzzy_y[i, 0], fuzzy_y[i, 1]))

            old_beta = np.copy(np.hstack((beta_prime_intercept, beta_prime_coef)))
            beta_prime_coef, beta_prime_intercept = self.fit_std_lr(self.X, curr_y)

            curr_it += 1

        return np.dot(beta_prime_coef, x) + beta_prime_intercept
