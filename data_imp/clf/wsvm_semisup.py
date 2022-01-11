from sklearn import preprocessing
import numpy as np
from sklearn.svm import SVC
import logging

from data_imp.clf.clf_meta import SemiSupervisedClassifier


class SemiSupWSVM(SemiSupervisedClassifier):
    """
    Weighted SVM for semi-supervised self learning experiment.
    """

    def __init__(self, C=1, num_w_iter=1, random_state=0):
        super().__init__(random_state)
        self.C = C
        self.num_w_iter = num_w_iter

    def fit(self, X_labeled, y_labeled, X_unlabeled):
        np.random.seed(self.random_state)

        self.scaler = preprocessing.StandardScaler().fit(np.vstack((X_labeled, X_unlabeled)))
        X_labeled = self.scaler.transform(X_labeled, copy=True)
        if X_unlabeled.shape[0] > 0:
            X_unlabeled = self.scaler.transform(X_unlabeled, copy=True)

        self.model = SVC(C=self.C, kernel="linear", max_iter=50000, random_state=self.random_state)
        self.model.fit(X_labeled, y_labeled)

        for it in range(self.num_w_iter):
            logging.debug("Starting iteration {}...".format(it))

            if X_unlabeled.shape[0] > 0:
                y_unlabeled = self.model.predict(X_unlabeled)
            else:
                y_unlabeled = np.array([])

            # Get weights based on distance to decision boundary
            inst_weights = np.ones([X_labeled.shape[0] + X_unlabeled.shape[0]])
            if X_unlabeled.shape[0] > 0:
                inst_weights[X_labeled.shape[0]:] = self.determine_semisup_weights(X_unlabeled, self.model)

            # Scale weights, s.t. highest value is 1
            self.model = SVC(C=self.C, kernel="linear", max_iter=50000, random_state=self.random_state)
            self.model.fit(np.vstack((X_labeled, X_unlabeled)), np.concatenate((y_labeled, y_unlabeled)),
                               sample_weight=inst_weights)
        return self

    def predict(self, X):
        X = self.scaler.transform(X, copy=True)
        return self.model.predict(X)
