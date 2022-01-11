import abc
import numpy as np
from scipy.optimize import minimize
import logging


class FLSVMMeta(abc.ABC):
    """
    Meta class providing the implementation of SVM-DI for both the conventional classifier and the semi-supervised model.
    """
    def __init__(self, C=1, max_iter=25, eps=1e-1, discrete_weights=False):
        self.C = C
        self.max_iter = max_iter
        self.eps = eps
        if discrete_weights is True or discrete_weights == "True":
            self.discrete_weights = True
        else:
            self.discrete_weights = False

    def _determine_instance_weights(self, X, init_clf, y):
        if self.discrete_weights:
            return np.where(np.abs(init_clf.decision_function(X) * y) <= 1., 1., 0.)
        else:
            # Determine k
            k = np.log(1. / (1-self.eps) - 1.)
            inst_weights = 1 / (1 + np.exp(k * init_clf.decision_function(X) * y))
            return inst_weights

    def _fit_cccp(self, X, y, coefs, inst_weights):
        X_full = np.hstack((np.ones([X.shape[0], 1]), X))

        theta = coefs.astype('float128')

        first = True
        iter = 0

        patience = 2
        patience_ctr = 0
        loss_cond = False

        loss_val = np.inf

        while first or (iter < self.max_iter and loss_cond):
            first = False

            initializer = coefs + np.random.rand(len(coefs)) * 0.01

            def min_fn_new(x):
                y_hat = np.dot(X_full, x)
                yyhat = y * y_hat

                vex = np.sum(np.where(yyhat < -1., np.maximum(0., 1. - yyhat) + (inst_weights - 1.) * (yyhat + 1.),
                                      np.maximum(0., 1. - yyhat)))

                cav_prime = np.zeros_like(x)
                for i in range(len(cav_prime)):
                    for l in range(X_full.shape[0]):
                        ys = y[l] * np.dot(X_full[l], theta)
                        if ys < 0:
                            cav_prime[i] += 2. * (1. - inst_weights[l]) * y[l] * X_full[l, i]

                return vex + cav_prime @ (x - theta)

            res = minimize(min_fn_new, initializer, method="Nelder-Mead")

            if res.fun == 0:
                theta = np.copy(res.x)
                break

            if res.fun > loss_val:
                patience_ctr += 1
            else:
                patience_ctr = 0

            loss_cond = patience_ctr < patience

            if loss_cond:
                theta_old = np.copy(theta)
                theta = np.copy(res.x)
            else:
                break

            if logging.getLogger().isEnabledFor(logging.DEBUG):
                logging.debug("New theta: {} | res fun: {}".format(theta, res.fun))

            loss_val = res.fun

            if logging.getLogger().isEnabledFor(logging.DEBUG):
                logging.debug("Weight difference: {}".format(np.linalg.norm(theta - theta_old)))
            iter += 1

        logging.debug("Needed {} iteration(s) within FLSVM optimization.".format(iter))

        return theta[0], theta[1:]