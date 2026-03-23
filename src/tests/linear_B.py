import matplotlib.pyplot as plt
import numpy as np

from solver import iterrarion_loop

# Tests PDE whos solution is W = 1 +3t
N = 30
# Space grid
r = np.linspace(0, 10, num=N)


# Time grid
t = np.linspace(0, 100, num=100)

# Initial value of W at t_min
W_init = np.array([1] * N)
# equation terms
D = np.array(
    [0.0001] * (N - 1)
)  # shall be zero but devision by 0 is a problem. Luckely 0.001 is kinda sorta 0.
S = np.array([3] * N)
V = np.array([1] * N)


def main():

    W = iterrarion_loop(r, t, W_init, D, S, V)
    renderer(r, t, W)


def renderer(r, t, W):
    # Rendering
    ax = plt.figure().add_subplot(projection="3d")
    a, b = np.meshgrid(r, t)
    ax.plot_surface(a, b, W)
    plt.show()


if __name__ == "__main__":
    main()
