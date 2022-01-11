from sklearn import svm, preprocessing
import numpy as np
from sklearn.svm import SVC

from data_imp.clf.clf_meta import Classifier


class OWSVM(Classifier):
    """
    One-step weighted SVM as proposed by Wu & Liu, 2013.
    """
    def __init__(self, C=1, random_state=0):
        super().__init__(random_state)
        self.C = C

    def fit(self, X, y):
        np.random.seed(self.random_state)

        self.scaler = preprocessing.StandardScaler().fit(X)
        X = self.scaler.transform(X, copy=True)

        classes = np.unique(y)
        n_features = X.shape[1]
        mc_problem = len(classes) > 2

        init_clf = SVC(C=self.C, kernel="linear", max_iter=50000, random_state=self.random_state)
        init_clf.fit(X, y)

        # Get weights based on distance to decision boundary
        inst_weights = np.zeros([X.shape[0]])
        for i in range(inst_weights.shape[0]):
            coefs = np.zeros(n_features + 1)

            # Get class of instance
            if mc_problem:
                class_idx = classes.index(y[i])
                coefs[1:] = init_clf.coef_[class_idx]
                coefs[0] = init_clf.intercept_[class_idx]
            else:
                # Binary class problem, i.e., a single decision boundary is given
                coefs[1:] = init_clf.coef_[0]
                coefs[0] = init_clf.intercept_[0]

            # Calculate the distance from point to decision boundary
            # inst_weights[i] = dist_line_to_point(coefs, X[i])
            inst_weights[i] = 1 / (1 + np.abs(init_clf.decision_function([X[i]])))

        self.final_clf =  SVC(C=self.C, kernel="linear", max_iter=50000, random_state=self.random_state)
        self.final_clf.fit(X, y, sample_weight=inst_weights)
        return self

    def predict(self, X):
        X = self.scaler.transform(X, copy=True)
        return self.final_clf.predict(X)
