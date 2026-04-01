import matplotlib.pyplot as plt
import numpy as np

from solver import iterrarion_loop

# Tests PDE whos solution is W(r,t) = 1
N = 30
tt = 1919
# Space grid
r = np.linspace(0, 10, num=N)
# Time grid
t = np.linspace(0, 10, num=tt)


# Initial value of W at t_min
W_init = np.array([1] * N)
# equation terms
D_0 = np.array(
    [0.0001] * N
)  # shall be zero but devision by 0 is a problem. Luckely 0.001 is kinda sorta 0.
S_0 = np.array([0] * N)
V_0 = np.array([1] * N)

bc_first = 0
bc_last = 0

D = np.repeat(D_0[np.newaxis, :], tt, axis=0)
print(D)
S = np.repeat(S_0[np.newaxis, :], tt, axis=0)
V = np.repeat(V_0[np.newaxis, :], tt, axis=0)


def main():
    W = iterrarion_loop(r, t, W_init, D, S, V, bc_first, bc_last)
    renderer(r, t, W)


def renderer(r, t, W):
    # Rendering
    ax = plt.figure().add_subplot(projection="3d")
    a, b = np.meshgrid(r, t)
    ax.plot_surface(a, b, W)
    plt.show()


if __name__ == "__main__":
    main()
