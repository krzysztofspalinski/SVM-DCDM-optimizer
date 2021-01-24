from time import sleep

import numpy as np

from optimizer import Optimizer


class DCDM(Optimizer):
    def __init__(self):
        pass

    def optimize(self, model, X, y, eps=10**-1):
        if model.w is None:
            model._Model__initialize_parameters(X.shape[1])
        U = model.C
        Q_conj_diag = np.einsum('ij,ji->i', X, X.T)
        alpha = np.zeros_like(y, dtype=float)
        iter_no = 0
        while iter_no < 10**3:  # WARUNEK OPTYMALNOŚCI
            iter_no += 1
            for i in np.random.permutation(range(len(alpha))):
                alpha_i = alpha[i]
                alpha_i_conj = alpha_i
                G = y[i] * np.matmul(X[i], model.w).item() - 1
                if alpha_i == 0:
                    PG = min(G, 0)
                elif alpha_i == U:
                    PG = max(G, 0)
                else:
                    PG = G
                if abs(PG) != 0:
                    alpha[i] = min(
                        max(alpha_i - G/Q_conj_diag[i], 0), U)
                    model.w += ((alpha[i] - alpha_i_conj) *
                                y[i] * X[i].T).reshape(-1, 1)
