import h5py
import matplotlib.pyplot as plt
import numpy as np

from solver import iterrarion_loop

file_path = "dream_outputs.h5"

with h5py.File(file_path, "r") as f:
    E_para = f["eqsys/E_field"][:]
    E_ceff = f["other/fluid/Eceff"][:]
    j_re = f["eqsys/j_re"][:]
    r = f["grid/r"][:]
    n_re = f["eqsys/n_re"][:]

N = 22
tt = 1919

# Calculate s from input data.
# E_para and j_re have shape (1919, 22) while E_ceff has shape (1918, 22
S = (E_para[0:-1] - E_ceff) * j_re[0:-1]
D = np.full((tt, N), 0.5)
V = np.full((tt, N), 1.0)


# grid
t = np.linspace(0, 0.2, tt)

# Initial value of W at t_min
W_init = (t[1] - t[0]) * (E_para[0, :] - E_ceff[0, :]) * j_re[0, :]

# These are the neumann conditions, and are therefore saying dW/dr=bc_first or bc_last
bc_first = 0
bc_last = 0


def main():
    W, W_avg_timeslice = iterrarion_loop(r, t, W_init, D, S, V, n_re, bc_first, bc_last)
    renderer_3d(r, t, W)
    renderer_2d(t, W_avg_timeslice)
    plt.show()


def renderer_3d(r, t, W):
    """Renders the energy matrix W with energy W[i,j] at radius r[i] and time t[j] in a 3d volumetric graf."""
    ax = plt.figure().add_subplot(projection="3d")
    a, b = np.meshgrid(r, t)

    plt.gca().set_xlabel("Radius (m)")
    plt.gca().set_ylabel("time (s)")
    plt.gca().set_zlabel("Energy (J)")
    plt.title("Simulated energy model")

    ax.plot_surface(a, b, W)


def renderer_2d(t, W_avg_timeslice):
    """Renders the radial avrige energy vektor W_avg_timeslice[i] over time t[i] as a graf."""
    plt.figure()
    plt.plot(t, W_avg_timeslice)
    plt.xlabel("Energy (J))")
    plt.xlabel("Time (s)")
    plt.ylabel("Average energy")
    plt.grid(True)


if __name__ == "__main__":
    main()
