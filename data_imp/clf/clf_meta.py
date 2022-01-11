from abc import ABC, abstractmethod
import numpy as np


class Classifier(ABC):
    """
    Meta class specifying the generic interface of a classifier.
    """
    def __init__(self, random_state=0):
        self.random_state = random_state

    @abstractmethod
    def fit(self, X, y):
        pass

    @abstractmethod
    def predict(self, X):
        pass


class SemiSupervisedClassifier(ABC):
    """
    Meta class specifying the generic interface of a semi-supervised classifier, which incorporates a basic weighting
    mechanism.
    """
    def __init__(self, random_state=0):
        self.random_state = random_state

    @abstractmethod
    def fit(self, X_labeled, y_labeled, X_unlabeled):
        pass

    @abstractmethod
    def predict(self, X):
        pass

    @staticmethod
    def determine_semisup_weights(X, clf):
        return 1 / (1 + np.abs(clf.decision_function(X)))
