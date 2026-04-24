import numpy as np


def iterrarion_loop(r, t, W_init, D, S, V, n_re, bc_first, bc_last):
    """Using finite volume method to solve a pde of the form
    (all partial derivatives) V(r)* dW/dt = d/dr(V(r)*D(r) * dW/dr ) + *V(r)*K(r).
    Parameters:
    r : np.ndarray
        Spatial grid; shape: lr
    t : np.ndarray
        Time grid; shape: lt
    W_init : np.ndarray
        Initial condition at t = 0, shape: lr
    D, S, V : np.ndarray
        Diffusion, source and flux surface coefficients; shape: lt*lr
    bc_first, bc_last: float
        These corespond to the Neuman boundary conditions at r=0 and r=max.
    Returns
    -------
    np.ndarray:
        A array of W filled with W(_r,_t) at position (r.index(_r), t.index(_t)).
    """
    lr = r.size
    lt = t.size
    print(str(lt) + "*" + str(lr))
    if lr != W_init.size:
        raise ValueError("'r' and '_W_init' have different shape")

    if D.shape != (lt, lr):
        raise ValueError(
            f"'D' must have shape lt*lr=({(lt, lr)}). Current shape is: " + str(D.shape)
        )
    if S.shape != (lt, lr):
        raise ValueError(
            f"'S' must have shape lt*lr=({(lt, lr)}). Current shape is: " + str(S.shape)
        )
    if V.shape != (lt, lr):
        raise ValueError(
            f"'V' must have shape lt*lr=({(lt, lr)}). Current shape is: " + str(V.shape)
        )
    d_r = np.diff(r)
    d_t = np.diff(t)
    # dr_volume is the width of a cell, while d_r is the width between centers
    dr_volume = np.zeros(lr)
    dr_volume[1:-1] = 0.5 * (d_r[:-1] + d_r[1:])
    dr_volume[0] = 0.5 * d_r[0]
    dr_volume[-1] = 0.5 * d_r[-1]

    # Initialise the solution matrix
    W = np.full((lt, lr), np.nan)
    W_init[0] = W_init[1] - bc_first * d_r[0]
    W_init[-1] = W_init[-2] + bc_last * d_r[-1]

    W[0, :] = W_init
    W[0] = bc_first
    W[-1] = bc_last

    W_avg_timeslice = np.zeros(lt)
    P = np.zeros((lr, lr))
    RHS = np.zeros(lr)
    for i in range(lt - 1):
        D_next = D[i + 1, :]
        S_next = S[i + 1, :]
        V_next = V[i + 1, :]

        VD = V_next * D_next
        VD_average = 2 / (1 / VD[0:-1] + 1 / VD[1:])

        A_left = VD_average[:-1] / (d_r[:-1] * dr_volume[1:-1])
        A_right = VD_average[1:] / (d_r[1:] * dr_volume[1:-1])
        A_center = V_next[1:-1] / d_t[i] + A_left + A_right

        P = (
            np.diag(np.concatenate(([0], A_center, [0])), k=0)
            + np.diag(np.concatenate(([0], -A_right)), k=1)
            + np.diag(np.concatenate((-A_left, [0])), k=-1)
        )

        RHS = V_next * S_next + (V_next * W[i, :]) / d_t[i]

        P[0, :] = 0
        P[0, 0] = -1
        P[0, 1] = 1
        RHS[0] = bc_first * d_r[0]

        P[-1, :] = 0
        P[-1, -2] = -1
        P[-1, -1] = 1
        RHS[-1] = bc_last * d_r[-1]
        W[i + 1, :] = np.linalg.solve(P, RHS)
        if np.isnan(W[i + 1, :]).any():
            raise ValueError("The resulting matrix contains NAN, should not happen!")

    W_total = np.sum(W * V * dr_volume, axis=1)
    N_total = np.sum(n_re[0:-1] * V * dr_volume, axis=1)
    W_avg_timeslice = W_total / N_total
    return W, W_avg_timeslice
