import numpy as np
from scipy.integrate import cumtrapz
import matplotlib.pyplot as plt


def remove_drift(array, touchdowns, fs, plot_drift=False, ylabel='Integrated value'):
    """
    Remove drift after integrating
    :param array: Integrated array to remove drift from [m/s]
    :param touchdowns: np.array with 0 where leg is in motion and 1 where leg is at the ground
    :param fs: sampling frequency [Hz]
    :param plot_drift: Plots effect of removing drift if true
    :param ylabel: Label to y axis of plot. Default value is "Integrated value".
    :return: vel: velocity array without drift [m/s]
    """
    # Set marker where motion start and stop
    diff = np.diff(np.asarray(touchdowns, dtype=int))
    start_motion = np.asarray(np.where(diff == -1)).flatten()  # Last index with zero value
    stop_motion = np.asarray(np.where(diff == 1)).flatten()  # Last index with non-zero value

    # Makes sure starting motion in the beginning, and stopping at the end
    if stop_motion[0] < start_motion[0]:
        start_motion = np.concatenate([[0], start_motion])

    if stop_motion[-1] < start_motion[-1]:
        stop_motion = np.concatenate([stop_motion, [len(array) - 1]])

    array_check = array.copy()
    # Remove drift
    for i in range(0, len(stop_motion)):
        # Find number of steps in the "motion"
        samples = stop_motion[i] - start_motion[i]

        # Find how much the drift is per sample vel[stop_motion[i]+1,:] == 0
        stepwise_drift = array[stop_motion[i], :] / samples

        # Find drift to remove from each step in the motion
        drift = [stepwise_drift * i for i in range(1, samples + 1)]

        # Remove drift
        array[start_motion[i] + 1:stop_motion[i] + 1, :] = array[start_motion[i] + 1:stop_motion[i] + 1, :] - drift

    # region Plot effect of removing drift
    if plot_drift:
        step_start = int((start_motion[5] + stop_motion[4]) / 2)
        step_stop = stop_motion[5]

        #fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3)
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, sharex=True)
        time = np.linspace(0, (step_stop-step_start) / fs, step_stop-step_start)

        ax1.set_title("x direction")
        ax1.plot(time, array_check[step_start:step_stop, 0], label="Before ")
        ax1.plot(time, array[step_start:step_stop, 0], label="After ")
        ax1.legend(loc='upper left')

        ax2.set_title("y direction")
        ax2.plot(time, array_check[step_start:step_stop, 1], label="Before")
        ax2.plot(time, array[step_start:step_stop, 1], label="After ")
        ax2.legend(loc='upper left')

        ax3.set_title("z direction")
        ax3.plot(time, array_check[step_start:step_stop, 2], label="Before ")
        ax3.plot(time, array[step_start:step_stop, 2], label="After ")
        ax3.legend(loc='upper left')
        '''
        ax4.set_title("Multiple steps, x direction")
        ax4.plot(array_check[0:step_stop, 0], label="Before ")
        ax4.plot(array[0:step_stop, 0], label="After ")
        ax4.legend(loc='upper left')

        ax5.set_title("Multiple steps, y direction")
        ax5.plot(array_check[0:step_stop, 1], label="Before")
        ax5.plot(array[0:step_stop, 1], label="After ")
        ax5.legend(loc='upper left')

        ax6.set_title("Multiple steps, z direction")
        ax6.plot(array_check[0:step_stop, 2], label="Before ")
        ax6.plot(array[0:step_stop, 2], label="After ")
        ax6.legend(loc='upper left')
        '''
        fig.text(0.5, 0.04, 'Time [s]', ha='center')
        fig.text(0.04, 0.5, ylabel, va='center', rotation='vertical')
        plt.show()

        plt.show()
    # endregion

    return array


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
