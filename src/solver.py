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

    # FIXME d_r[0] is lazy solution
    # FIXME use neuman conditions
    for i in range(lt - 1):
        """
        Need to change this since we are supposed to take the harmonic average over V*D not just D
        D_average = 2 / (1 / D[0:-1] + 1 / D[1:])
        """
        # take next values, for euler backwards
        # this is correct, we take the next value in time.
        # Is diffusivity given at faces or taken at faces? Do we know the diffusivity of a cell, or through the cell to its neighbours?
        # FIXME
        D_next = D[i + 1, :]
        S_next = S[i + 1, :]
        V_next = V[i + 1, :]
        # VD is taken at a face now
        VD = V_next * D_next
        VD_average = 2 / (1 / VD[0:-1] + 1 / VD[1:])

        P = np.zeros((lr, lr))
        RHS = np.zeros(lr)
        P[0, 0] = 1
        P[-1, -1] = 1
        RHS[0] = bc_first
        RHS[-1] = bc_last

        # Now technically, I shouldn't have **2 for a finite volume scheme.
        # But since everything is assumed to be uniform, it is fine. If we use DREAM inputs, then this has to be changed
        # FIXME maybe fixed?
        for j in range(1, lr - 1):
            a_left = VD_average[j - 1] / (d_r[j - 1])
            a_right = VD_average[j] / (d_r[j])

            P[j, j - 1] = -a_left
            P[j, j] = (V_next[j] * dr_volume[j]) / d_t[i] + a_left + a_right
            P[j, j + 1] = -a_right

            # Calculate the rhs based on current cell
            RHS[j] = V_next[j] * dr_volume[j] * (S_next[j] + W[i, j] / d_t[i])

        # Do some reasonable checks
        if np.isnan(P).any():
            print("error here in P")

        W[i + 1, :] = np.linalg.solve(P, RHS)

        # Do other reasonable checks
        if np.isnan(W[i + 1, :]).any():
            print("error here in W")  # RaiseError!!
    return W
