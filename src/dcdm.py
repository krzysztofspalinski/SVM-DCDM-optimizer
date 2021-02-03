from time import sleep

import numpy as np

from optimizer import Optimizer


class DCDM(Optimizer):
    def __init__(self):
        pass

    def optimize(self, model, X, y, P, q, eps=5*10**-1):
        if model.w is None:
            model._Model__initialize_parameters(X.shape[1])
        U = model.C
        Q_conj_diag = np.einsum('ij,ji->i', X, X.T)
        alpha = np.zeros_like(y, dtype=float)
        iter_no = 0
        loss_history_gain = np.full(5, np.inf)
        columns = ["iteration", "pcost", "dcost", "pres", "dres"]
        print(" ".join([f"{c:>12}" for c in columns]))
        old_pcost = model.compute_obj_primal(alpha, P, q)
        values = [iter_no, old_pcost, "na", "na", "na"]
        print(" ".join([f"{v}" for v in values]))
        while True:
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
            new_pcost = model.compute_obj_primal(alpha, P, q)
            values = [iter_no, new_pcost, "na", "na", "na"]
            print(" ".join([f"{v:>12}" for v in values]))
            loss_history_gain[iter_no % 5] = np.abs(old_pcost - new_pcost)
            if np.std(loss_history_gain)/np.max(loss_history_gain+eps) < eps:
                return
            else:
                old_pcost = new_pcost
