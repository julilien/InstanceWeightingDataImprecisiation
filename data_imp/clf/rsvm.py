import cvxpy as cvx
import numpy as np
from sklearn import preprocessing
import logging

from data_imp.clf.clf_meta import Classifier


class RSVM(Classifier):
    """
    Implementation of the robust SVM as described in "Trading Convexity for Scalability" (Collobert et. al., 2006).
    """
    def __init__(self, s=-1, C=1, num_iter=100, random_state=0):
        super().__init__(random_state)

        self.s = s
        self.C = C
        self.num_iter = num_iter

    @staticmethod
    def apply_f(x, W, b):
        return np.dot(W, x) + b

    def fit(self, X, y):
        np.random.seed(self.random_state)

        self.scaler = preprocessing.StandardScaler().fit(X)
        X = self.scaler.transform(X, copy=True)

        self.X = X

        n_insts = X.shape[0]
        alpha = cvx.Variable(n_insts)
        w_alpha = np.zeros(n_insts)
        beta = np.zeros(n_insts)
        b_t = 0

        # Init beta
        beta_old = np.copy(beta)

        K = np.zeros([n_insts, n_insts])
        for i in range(K.shape[0]):
            for j in range(K.shape[1]):
                K[i, j] = y[i] * y[j] * (X[i].T @ X[j])
        first_run = True
        it_ctr = 0

        # Implementation of CCCP for ramp loss SVM
        while first_run or np.linalg.norm(beta_old - beta) > 1e-3:
            if it_ctr > self.num_iter:
                logging.debug("The number of maximal iterations ({}) is reached. Aborting optimization...".format(self.num_iter))
                break

            first_run = False

            aKa = cvx.quad_form(alpha, K)

            expr = cvx.sum(alpha) - 0.5 * aKa

            constraints = [-beta_old <= alpha]
            constraints += [alpha <= self.C - beta_old]
            constraints.append(y @ alpha == 0)
            max_prob = cvx.Problem(cvx.Maximize(expr), constraints)
            try:
                max_prob.solve('CVXOPT') # 'OSQP'
            except cvx.SolverError:
                # Could not solve using CVXOPT, trying SCS (should always return a solution)
                logging.warning("Got solver error. Trying to apply SCS.")
                max_prob.solve('SCS', max_iters=20000)
            w_alpha = alpha.value

            sv_ctr = 0
            def f(query_x, x, y, alpha, b):
                return np.sum(y * alpha * np.dot(x, query_x)) + b

            for i in range(n_insts):
                if 0 < w_alpha[i] < self.C:
                    b_t += y[i] - (f(X[i], X, y, w_alpha, b_t) - b_t)
                    sv_ctr += 1
            b_t /= sv_ctr

            beta_old = np.copy(beta)
            for i in range(n_insts):
                if f(X[i], X, y, w_alpha, b_t) < self.s:
                    beta[i] = self.C
                else:
                    beta[i] = 0

            it_ctr += 1

        # Final parameters
        self.w = y * w_alpha
        self.b = b_t

        return self

    def predict_single_instance(self, x):
        if (np.sum(self.w * np.dot(self.X,x)) + self.b) >= 0:
            return +1
        else:
            return -1

    def predict(self, x):
        if not isinstance(x, np.ndarray):
            x = np.array(x)

        if len(x.shape) == 1:
            x = np.expand_dims(x, axis=0)

        x = self.scaler.transform(x, copy=True)

        preds = []
        for i in range(x.shape[0]):
            preds.append(self.predict_single_instance(x[i]))
        return preds