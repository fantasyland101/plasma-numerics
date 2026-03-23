import matplotlib.pyplot as plt
import numpy as np
import h5py

from solver import iterrarion_loop

file_path = 'dream_outputs.h5'

with h5py.File(file_path, 'r') as f:
    E_para = f['eqsys/E_field'][:]
    E_ceff = f['other/fluid/Eceff'][:]
    j_re = f['eqsys/j_re'][:]

N = 22
tt = 1919

# Calculate s from input data.
# E_para and j_re have shape (1919, 22) while E_ceff has shape (1918, 22)
S = (E_para[0:-1] - E_ceff) * j_re[0:-1]
D = np.array([3] * N)
V = np.array([3] * N)
print(S.shape)

# Space grid
r = np.linspace(0, 3, N)

# Time grid
t = np.linspace(0, 3, tt)


# Initial value of W at t_min
W_init = np.array([1] * N)
bc_first = 1
bc_last = 1
W_init[0] = bc_first
W_init[-1] = bc_last



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
