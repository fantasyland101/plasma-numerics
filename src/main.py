import matplotlib.pyplot as plt
import numpy as np
import h5py

from solver import iterrarion_loop

file_path = 'dream_outputs.h5'

with h5py.File(file_path, 'r') as f:
    E_para = f['eqsys/E_field'][:]
    E_ceff = f['other/fluid/Eceff'][:]
    j_re = f['eqsys/j_re'][:]
    r = f['grid/r'][:]
    n_re = f['eqsys/n_re'][:]

N = 22
tt = 1919

# Calculate s from input data.
# E_para and j_re have shape (1919, 22) while E_ceff has shape (1918, 22
S = (E_para[0:-1] - E_ceff) * j_re[0:-1]
D = np.full((tt, N), 0.5)
V = np.full((tt, N), 1.0)


# grid
t = np.linspace(0, 0.2, tt)
d_r = np.diff(r)
d_t = np.diff(t)

# Initial value of W at t_min
W_init = d_t[0] * (E_para[0, :] - E_ceff[0, :]) * j_re[0, :]

# These are the neumann conditions, and are therefore saying dW/dr=bc_first or bc_last
bc_first = 0
bc_last = 0


def main():
    W, W_avg_timeslice = iterrarion_loop(r, t, d_r, d_t, W_init, D, S, V, n_re, bc_first, bc_last)
    renderer(r, t, W)
    renderer2(t, W_avg_timeslice)
    plt.show()


def renderer(r, t, W):
    # Rendering
    ax = plt.figure().add_subplot(projection="3d")
    a, b = np.meshgrid(r, t)
    ax.plot_surface(a, b, W)
# my lazy fix becaus
def renderer2(t, W_avg_timeslice):
    plt.figure()
    plt.plot(t, W_avg_timeslice)

    plt.xlabel("Time")
    plt.ylabel("Average energy")
    plt.grid(True)

if __name__ == "__main__":
    main()