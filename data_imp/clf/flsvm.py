import numpy as np
from sklearn import svm, preprocessing
from sklearn.svm import SVC

from data_imp.clf.clf_meta import Classifier
from data_imp.clf.flsvm_meta import FLSVMMeta

import logging


class FLSVM(Classifier, FLSVMMeta):
    """
    Our SVM-DI classifier implementation.
    """
    def __init__(self, C=1, random_state=0, max_iter=25, eps=1e-1, discrete_weights=False):
        Classifier.__init__(self, random_state=random_state)
        FLSVMMeta.__init__(self, C=C, max_iter=max_iter, eps=eps, discrete_weights=discrete_weights)

    def fit(self, X, y):
        np.random.seed(self.random_state)

        self.scaler = preprocessing.StandardScaler().fit(X)
        X = self.scaler.transform(X, copy=True)

        classes = np.unique(y)
        if len(classes) > 2:
            # Currently, only binary class problems are supported
            raise NotImplementedError

        n_features = X.shape[1]

        init_clf = SVC(C=self.C, kernel="linear", max_iter=50000, random_state=self.random_state)
        init_clf.fit(X, y)

        if logging.getLogger().isEnabledFor(logging.DEBUG):
            logging.debug("Initial theta: {}, {}".format(init_clf.coef_, init_clf.intercept_))

        coefs = np.zeros(n_features + 1)
        coefs[1:] = init_clf.coef_[0]
        coefs[0] = init_clf.intercept_[0]

        # Get weights based on distance to decision boundary
        inst_weights = self._determine_instance_weights(X, init_clf, y)

        coef0, coefs = self._fit_cccp(X, y, coefs, inst_weights)

        self.final_clf = svm.LinearSVC(C=self.C, random_state=self.random_state)
        self.final_clf.coef_ = coefs.reshape(1, -1)
        self.final_clf.intercept_ = coef0
        self.final_clf.classes_ = np.unique(y)
        return self

    def predict(self, X):
        X = self.scaler.transform(X, copy=True)

        return self.final_clf.predict(X)
