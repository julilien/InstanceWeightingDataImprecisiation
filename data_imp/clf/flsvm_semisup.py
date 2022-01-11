import numpy as np
from sklearn import svm, preprocessing
from sklearn.svm import SVC
import logging

from data_imp.clf.clf_meta import SemiSupervisedClassifier
from data_imp.clf.flsvm_meta import FLSVMMeta


class SemiSupFLSVM(SemiSupervisedClassifier,FLSVMMeta):
    """
    The semi-supervised version of our SVM-DI model.
    """
    def __init__(self, C=1, random_state=0, max_iter=25, eps=1e-1, discrete_weights=False, num_w_iter=1):
        SemiSupervisedClassifier.__init__(self, random_state=random_state)
        FLSVMMeta.__init__(self, C=C, max_iter=max_iter, eps=eps, discrete_weights=discrete_weights)
        self.num_w_iter = num_w_iter

    def fit(self, X_labeled, y_labeled, X_unlabeled):
        np.random.seed(self.random_state)

        self.scaler = preprocessing.StandardScaler().fit(np.vstack((X_labeled, X_unlabeled)))
        X_labeled = self.scaler.transform(X_labeled, copy=True)
        if X_unlabeled.shape[0] > 0:
            X_unlabeled = self.scaler.transform(X_unlabeled, copy=True)

        classes = np.unique(y_labeled)
        if len(classes) > 2:
            raise NotImplementedError

        n_features = X_labeled.shape[1]

        init_model = SVC(C=self.C, kernel="linear", max_iter=50000, random_state=self.random_state)
        init_model.fit(X_labeled, y_labeled)

        inst_weights = np.ones([X_labeled.shape[0] + X_unlabeled.shape[0]])
        inst_weights[X_labeled.shape[0]:] = np.zeros(X_unlabeled.shape[0])

        coefs = np.zeros(n_features + 1)
        coefs[1:] = init_model.coef_[0]
        coefs[0] = init_model.intercept_[0]

        coef0, coefs = self._fit_cccp(np.vstack((X_labeled, X_unlabeled)),
                                      np.concatenate((y_labeled, np.ones(X_unlabeled.shape[0]))),
                                      coefs, inst_weights)

        self.model = svm.LinearSVC(C=self.C, random_state=self.random_state)
        self.model.coef_ = coefs.reshape(1, -1)
        self.model.intercept_ = coef0
        self.model.classes_ = np.unique(y_labeled)

        if logging.getLogger().isEnabledFor(logging.DEBUG):
            logging.debug("Initial theta: {}, {}".format(self.model.coef_, self.model.intercept_))

        for it in range(self.num_w_iter):
            logging.debug("Starting iteration {}...".format(it))

            coefs = np.zeros(n_features + 1)

            coefs[1:] = self.model.coef_[0]
            if type(self.model.intercept_) is list or type(self.model.intercept_) is np.ndarray:
                coefs[0] = self.model.intercept_[0]
            else:
                coefs[0] = self.model.intercept_

            if X_unlabeled.shape[0] > 0:
                y_unlabeled = self.model.predict(X_unlabeled)
            else:
                y_unlabeled = np.array([])

            # Get weights based on distance to decision boundary
            inst_weights = np.ones([X_labeled.shape[0] + X_unlabeled.shape[0]])
            if X_unlabeled.shape[0] > 0:
                inst_weights[X_labeled.shape[0]:] = self.determine_semisup_weights(X_unlabeled, self.model)

            coef0, coefs = self._fit_cccp(np.vstack((X_labeled, X_unlabeled)), np.concatenate((y_labeled, y_unlabeled)),
                                          coefs, inst_weights)

            self.model = svm.LinearSVC(C=self.C, random_state=self.random_state)
            self.model.coef_ = coefs.reshape(1, -1)
            self.model.intercept_ = coef0
            self.model.classes_ = np.unique(y_labeled)
        return self

    def predict(self, X):
        X = self.scaler.transform(X, copy=True)

        return self.model.predict(X)
