from sklearn import preprocessing
from sklearn.svm import SVC

from data_imp.clf.clf_meta import Classifier

import numpy as np


class SVMWrapper(Classifier):
    """
    Wrapper of a conventional, unregularized SVM.
    """
    def __init__(self, C=1, random_state=0):
        super().__init__(random_state)
        self.C = C

    def fit(self, X, y):
        np.random.seed(self.random_state)

        self.scaler = preprocessing.StandardScaler().fit(X)
        X = self.scaler.transform(X, copy=True)

        self.model = SVC(C=self.C, kernel="linear", max_iter=50000, random_state=self.random_state)
        self.model.fit(X, y)
        return self

    def predict(self, X):
        X = self.scaler.transform(X, copy=True)
        return self.model.predict(X)