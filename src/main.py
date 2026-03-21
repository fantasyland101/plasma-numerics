import matplotlib.pyplot as plt
import numpy as np

from solver import iterrarion_loop

# Note from hugo: If too slow use more numpy
# Note I did wrie the mesh to be more general like lorenze did!

# Space grid
r = np.linspace(0, 3, num=10)

# Time grid
t = np.linspace(0, 3, num=10)


# Initial value of W at t_min
W_init = np.array([1] * 10)
bc_first = 1
bc_last = 1
W_init[0] = bc_first
W_init[-1] = bc_last

# equation terms
D = np.array([0.1] * 10)
S = np.array([4] * 10)
V = np.array([3] * 10)


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
