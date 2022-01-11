from abc import ABC, abstractmethod
import numpy as np
from sklearn import preprocessing


class Regressor(ABC):
    """
    Meta class specifying a general interface for regressors.
    """
    def __init__(self, scaling=True, random_state=0):
        self.scaling = scaling
        self.random_state = random_state

    def fit_scaler(self, X):
        if self.scaling:
            self.scaler = preprocessing.StandardScaler().fit(X)
            X = self.scaler.transform(X, copy=True)
        return X

    def scale_features(self, x):
        if self.scaling:
            undo_reshape = False
            if x.ndim == 1:
                undo_reshape = True
                x = x.reshape(1, -1)
            x = self.scaler.transform(x, copy=True)

            if undo_reshape:
                x = x.reshape(-1)
        return x

    @abstractmethod
    def fit(self, X, y):
        pass

    @abstractmethod
    def predict_single_instance(self, x):
        pass

    def predict(self, X):
        if len(X.shape) < 2:
            X = np.expand_dims(X, axis=0)

        predictions = []
        for i in range(X.shape[0]):
            predictions.append(self.predict_single_instance(X[i]))

        return predictions
