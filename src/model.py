import gc
from time import sleep, time

import numpy as np
import qpsolvers
from cvxpy.error import DCPError
from sklearn import datasets
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from dcdm import DCDM
from optimizer import Optimizer
from utils import Capturing, process_output

OPTIMIZERS = [
    DCDM,
    # 'cvxopt',
    # 'ecos', SLOW
    # 'quadprog', SLOW
    # 'osqp',
    # 'cvxpy_GUROBI', SLOW
    'cvxpy_MOSEK',
    'cvxpy_SCS',
    # === 'gurobi', NOT OPTIMAL
    # === 'mosek',  # NOT OPTIMAL
    # === 'cvxpy_CPLEX', NOT AVAILABLE
    # === 'cvxpy_OSQP', LITTLE DIFFERENCE WITH osqp
    # === 'cvxpy_ELEMENTAL', NOT AVAILABLE
    # === 'cvxpy_ECOS', NOT AVAILABLE
    # === 'cvxpy_ECOS_BB', NOT AVAILABLE
    # === 'cvxpy_CVXOPT', NOT AVAILABLE
]


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
            eps = 1e-5
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
            start = time()
            with Capturing() as output:
                if optimizer.startswith('cvxpy_'):
                    alphas = qpsolvers.cvxpy_solve_qp(
                        P, q, G, h, A, b, solver=optimizer[6:], verbose=True)
                else:
                    alphas = qpsolvers.solve_qp(
                        P, q, G, h, A, b, solver=optimizer, verbose=True)
            runtime1 = time() - start

            result, result_df, runtime2 = process_output(output, optimizer)
            if result:
                w = np.matmul(alphas, X_dash)
                self.w = w
            else:
                print("Solve process failed.")
            return result, result_df, runtime2 or runtime1
        elif issubclass(optimizer, Optimizer):
            eps = 1e-5
            m, n = X.shape
            X_dash = y.reshape(-1, 1) * X
            P = np.dot(X_dash, X_dash.T) * 1.
            P += np.eye(*P.shape) * eps
            q = -np.ones((m, 1)).reshape((m,))
            opt = optimizer()
            start = time()
            with Capturing() as output:
                opt.optimize(self, X, y, P, q)
            runtime1 = time() - start
            result, result_df, runtime2 = process_output(output, optimizer)
            return result, result_df, runtime2 or runtime1
        else:
            raise ValueError(f'Optimizer {optimizer} is not recognized')

    def compute_gradient(self, X, y):
        loss = self.compute_loss(X, y)
        gradient = np.zeros_like(loss)
        gradient[loss < 0] = -1
        return gradient

    def compute_obj_primal(self, alphas, P, q):
        val = (1/2) * alphas.T @ P @ alphas + q.T @ alphas
        return val.item()

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
        n_samples=100, cluster_std=2, centers=2, n_features=500)
    y[y == 0] = -1
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    for optimizer in OPTIMIZERS:
        print("#" * 40, f"Optimizer: {optimizer}", sep="\n")
        try:
            m = Model()
            result, result_df, duration = m.fit(
                X_train, y_train, optimizer=optimizer)
            print(f'Solved: {result}')
            print(result_df)
            print(f'Accuracy: {accuracy_score(m.predict(X_test), y_test)}',
                  f"Czas: {duration}", "", sep="\n")
        except (DCPError, ValueError) as e:
            print(f"Błąd: {e}", "", sep="\n")
            raise e
