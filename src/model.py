from time import time

import numpy as np
import qpsolvers
from cvxpy.error import DCPError
from sklearn import datasets
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from dcdm import DCDM
from optimizer import Optimizer
import gc

from utils import Capturing, process_output

OPTIMIZERS = [DCDM, 'cvxpy', 'cvxopt', 'quadprog']


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
        if type(optimizer) == str:
            eps = 1e-10
            m, n = X.shape
            y = y.reshape(-1, 1) * 1.
            X_dash = y * X
            P = np.dot(X_dash, X_dash.T) * 1.
            P += np.eye(*P.shape) * eps
            q = -np.ones((m, 1)).reshape((m,))
            G = -np.eye(m)
            G += np.eye(*G.shape) * eps
            h = np.zeros(m).reshape((m,))
            A = y.reshape(1, -1)
            b = np.zeros(1)

            gc.collect()
            with Capturing() as output:
                alphas = qpsolvers.solve_qp(P, q, G, h, A, b, solver=optimizer, verbose=True)

            solve_succeeded, result_df = process_output(output)

            if solve_succeeded:
                w = np.matmul(alphas, X_dash)
                self.w = w
                return result_df
            else:
                print("Solve process failed.")
                print(f"Reason: \"{output[-1]}\"")
                return result_df
        elif issubclass(optimizer, Optimizer):
            opt = optimizer()
            opt.optimize(self, X, y)
        else:
            raise ValueError(f'Optimizer {optimizer} is not recognized')

    def compute_gradient(self, X, y):
        loss = self.compute_loss(X, y)
        gradient = np.zeros_like(loss)
        gradient[loss < 0] = -1
        return gradient

    def compute_loss(self, X, y):
        loss = np.sum(np.maximum(
            1 - y * np.matmul(X, self.w),
            0
        ))
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


if __name__ == "__main__":
    X, y = datasets.make_blobs(
        n_samples=1000, cluster_std=12, centers=2, n_features=5000)
    y[y == 0] = -1
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    for optimizer in OPTIMIZERS:
        try:
            m = Model()
            start = time()
            m.fit(X_train, y_train, optimizer=optimizer)
            print("#" * 40, f"Optimizer: {optimizer}", accuracy_score(
                m.predict(X_test), y_test), f"Czas: {time() - start}", "", sep="\n")
        except (DCPError, ValueError):
            print("#" * 40, f"Optimizer: {optimizer}", "Błąd", "", sep="\n")
