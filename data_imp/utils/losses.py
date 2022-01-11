from abc import ABC, abstractmethod


class Loss(ABC):
    @abstractmethod
    def loss(self, y1, y2):
        pass


class SquaredErrorLoss(Loss):
    def loss(self, y1, y2):
        return (y1 - y2) ** 2
