import numpy as np
from node.position import remove_drift
from scipy.integrate import cumtrapz


def find_velocity(acc, touchdowns, fs, plot_drift=False):
    """
    Integrate acceleration to get velocity. Removes drift after integration
    :param acc: acceleration [m/s/s]
    :param touchdowns: np.array with 0 where leg is in motion and 1 where leg is at the ground
    :param fs: sampling frequency [Hz]
    :param plot_drift: Plots effect of removing drift if true
    :return: vel: velocity [m/s]
    """
    vel = np.zeros((len(acc), 3))

    # Integrate acceleration
    for i in range(1, len(acc)):
        v0 = np.asarray(vel[i - 1, :])
        a = np.asarray(acc[i, :])
        vel[i, :] = v0 + a * 1 / fs

        # Force vel to be zero when foot is touching ground
        if touchdowns[i] != 0:
            vel[i, :] = [0, 0, 0]

    # Remove drift
    vel = remove_drift(vel, touchdowns, plot_drift)

    return vel


def find_velocity_cumtrapz(acc, fs):
    vel = cumtrapz(acc, dx=1 / fs)
    return vel
