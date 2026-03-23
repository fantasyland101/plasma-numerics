import matplotlib.pyplot as plt
import numpy as np

from solver import iterrarion_loop

# Tests PDE whos solution is as such:
# W(r, t= infinity) = -(S*r^2) / (2D) + (S*r)/(2D)

N = 100
# Space grid
r = np.linspace(0, 1, num=N)
# Time grid
t = np.linspace(0, 100, num=100)

# Initial value of W at t_min
W_init = np.array([0] * N)
# equation terms
D = np.array(
    [1] * (N)
)  # shall be zero but devision by 0 is a problem. Luckely 0.001 is kinda sorta 0.
S = np.array([3] * N)
V = np.array([1] * N)

bc_first = 0
bc_last = 0
W_init[0] = bc_first
W_init[-1] = bc_last

def main():
    W = iterrarion_loop(r, t, W_init, D, S, V, bc_first, bc_last)
    renderer(r, t, W)


def renderer(r, t, W):
    # solution plot to reference as t goes to infinity
    # CA and CB is not correct values here yet
    _S = 3
    _D = 1
    solution = [-(_S * _r**2) / (2 * _D) + (_S * _r) / (2 * _D) for _r in r]

    _, ax = plt.subplots()
    for i in range(len(t) - 1):
        ax.cla()
        ax.plot(r, W[i, :])
        ax.plot(r, solution)
        plt.pause(1)  # change time here to make the timesteps take longer
    input("done. press to stop")


if __name__ == "__main__":
    main()
