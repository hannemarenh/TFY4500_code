import numpy as np
from node.position import remove_drift
from scipy.integrate import cumtrapz
import matplotlib.pyplot as plt


def plot_velocity(vel, fs):
    """
    Plots velocity as func of sample number
    :param vel: Velocity with dim (measurements, 3)
    :param fs: sampling frequency [Hz]
    :return: 0
    """
    time = np.linspace(0, len(vel)/fs, len(vel))

    plt.figure()
    plt.plot(time, vel[:, 0], label="x direction")
    plt.plot(time, vel[:, 1], label="y direction")
    plt.plot(time, vel[:, 2], label="z direction")
    plt.legend(loc='upper left')
    plt.ylabel("Velocity [m/s]")
    plt.xlabel("Time [s]")
    plt.show()

    return 0


def find_velocity(acc, touchdowns, fs, plot_drift=False, plot_vel=False):
    """
    Integrate acceleration to get velocity. Removes drift after integration
    :param acc: acceleration [m/s/s]
    :param touchdowns: np.array with 0 where leg is in motion and 1 where leg is at the ground
    :param fs: sampling frequency [Hz]
    :param plot_drift: Plots effect of removing drift if true
    :param plot_vel: Plots resulting velocity if true

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
    vel = remove_drift(vel, touchdowns, fs, plot_drift, 'Velocity [m/s]')

    if plot_vel:
        plot_velocity(vel, fs)

    return vel


def find_velocity_cumtrapz(acc, fs, plot_vel=False):
    """
    Integrate acceleration to get velocity. Removes drift after integration
    :param acc: acceleration [m/s/s]
    :param touchdowns: np.array with 0 where leg is in motion and 1 where leg is at the ground
    :param fs: sampling frequency [Hz]
    :param plot_drift: Plots effect of removing drift if true
    :param plot_vel: Plots resulting velocity if true
    :return: vel: velocity [m/s]
    """
    vel = np.zeros(acc.shape)
    vel[:, 0] = cumtrapz(acc[:, 0], dx=1 / fs, initial=0)
    vel[:, 1] = cumtrapz(acc[:, 1], dx=1 / fs, initial=0)
    vel[:, 2] = cumtrapz(acc[:, 2], dx=1 / fs, initial=0)

    if plot_vel:
        plot_velocity(vel, fs)
    return vel
