import numpy as np
from sklearn import datasets

from dcdm import DCDM


class Model:

    def __init__(self, C=1, w=None):
        self.w: np.ndarray = w
        self.C: float = C

    def predict(self, X):
        if self.w is None:
            raise AttributeError('Model parameters (w) are not set.')
        y_hat_proba = np.matmul(X, self.w)
        y_hat = np.zeros_like(y_hat_proba)
        y_hat[y_hat_proba > 0] = 1
        y_hat[y_hat_proba <= 0] = -1
        return y_hat.ravel()

    def fit(self, X, y, optimizer=DCDM):
        opt = optimizer()
        opt.optimize(self, X, y)

    def compute_gradient(self, X, y):
        loss = self.compute_loss(X, y)
        gradient = np.zeros_like(loss)
        gradient[loss < 0] = -1
        return gradient

    def compute_loss(self, X, y):
        loss = np.maximum(
            1 - y * np.matmul(X, self.w),
            0
        )
        return loss

    def __initialize_parameters(self, m):
        """
        Initializes model parameters
        :param m: dataset dimension
        """
        if self.w is not None and self.w.shape[0] == m:
            return
        else:
            self.w = np.zeros((m, 1))
