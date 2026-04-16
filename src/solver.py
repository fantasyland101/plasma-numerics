import numpy as np


def iterrarion_loop(r, t, W_init, D, S, V, bc_first, bc_last):
    """Using finite volume method to solve a pde of the form
    (all partial derivatives) V(r)* dW/dt = d/dr(V(r)*D(r) * dW/dr ) + *V(r)*K(r).
    Parameters:
    r : np.ndarray
        Spatial grid; shape: lr
    t : np.ndarray
        Time grid; shape: lt
    W_init : np.ndarray
        Initial condition at t = 0, shape: lr
        First and last values are Dirichlet boundary values
    D, S, V : np.ndarray
        Diffusion, source and flux surface coefficients.
        These can be passed as either a list (time-independent) or an array (time-dependent)
        The function make_time_dependent takes a coefficient and pastes it to make a lr*lt array

    Returns
    -------
    np.ndarray:
        A array of W filled with W(_r,_t) at position (r.index(_r), t.index(_t)).
    """
    lr = r.size
    lt = t.size
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

    # callculate the step lengths
    # FIXME "d_r is distances betwean flux surfaces and should be given as input params to this function"
    # is d_r really the distance between faces? I don't think so, it should be the distance between centers.
    d_r = np.diff(r)
    d_t = np.diff(t)

    # dr_volume is the width of a cell, while d_r is the width between centers
    dr_volume = np.zeros(lr)
    dr_volume[1:-1] = 0.5 * (d_r[:-1] + d_r[1:])
    dr_volume[0] = 0.5 * d_r[0]
    dr_volume[-1] = 0.5 * d_r[-1]

    # Initilise the kinetic energy matrix
    W_size = len(t), len(r)
    W = np.full(W_size, np.nan)
    W[0, :] = W_init
    W[0] = bc_first
    W[-1] = bc_last

    P = np.zeros((lr, lr))
    RHS = np.zeros(lr)
    # FIXME d_r[0] is lazy solution
    # FIXME use neuman conditions
    for i in range(lt - 1):
        # take next values, for euler backwards
        # this is correct, we take the next value in time.
        # Is diffusivity given at faces or taken at faces? Do we know the diffusivity of a cell, or through the cell to its neighbours?
        # FIXME
        D_now = D[i, :]
        S_now = S[i, :]
        V_now = V[i, :]
        # VD is taken at a face now
        VD_average = 2 / (1 / (V_now * D_now)[0:-1] + 1 / (V_now * D_now)[1:])

        # Calculate the time step matrix P
        # P[j,j] = A_center[j] for j = [1, max -1]
        # P[max,max] = 1 (boundary cond)
        # P[0,0] = 1 (boundary cond)
        # P[j, j+1] = -A_right[j] for j = [1, max]
        # P[0,1] =0 (boundary cond)
        # P[j,j-1] = -A_left[j] for j =[0, max -1]
        # P[max, max-1] =0 (boundary cond)
        # A_center = ..
        # A_left = ..
        # A_right = ..
        T = -d_t[i] * VD_average[1:] / (d_r[1:] * dr_volume[1:-1] * V_now[1:-1])
        B = -d_t[i] * VD_average[:-1] / (d_r[:-1] * dr_volume[1:-1] * V_now[1:-1])
        C = 1 - T - B
        P = (
            np.diag(np.concatenate(([0], C, [0])), k=0)
            + np.diag(np.concatenate(([0], T)), k=1)
            + np.diag(np.concatenate((B, [0])), k=-1)
        )
        ## Set boundary conditions. for only dw/dr =0 at boundary
        K_l = d_r[0]
        K_r = d_r[-1]
        P[0, 0] = B[0]
        P[0, 1] = C[0]
        P[0, 2] = T[0]

        P[-1, -3] = B[-1]
        P[-1, -2] = C[-1]
        P[-1, -1] = T[-1]

        # Calculate the RHS of the equation
        RHS = d_t[i] * S_now + W[i, :]
        RHS[0] = d_t[i] * S_now[1] + K_l + W[i, 0]
        RHS[-1] = d_t[i] * S_now[-2] + K_r + W[i, -1]

        print(P)

        # Do some reasonable checks
        if np.isnan(P).any():
            print("error here in P")

        W[i + 1, :] = np.linalg.solve(P, RHS)

        if np.isnan(W[i + 1, :]).any():
            raise ValueError("The resulting matrix contains NAN, should not happen!")
    return W
