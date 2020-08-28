# region import packages
import numpy as np
import pandas as pd
from node.rotation import rotation_matrix_from_vectors
from scipy import integrate
import matplotlib.pyplot as plt
from scipy.fft import fft


# endregion


def remove_drift(array, touchdowns, plot_drift=False):
    """
    Remove drift after integrating
    :param array: Integrated array to remove drift from [m/s]
    :param touchdowns: np.array with 0 where leg is in motion and 1 where leg is at the ground
    :param plot_drift: Plots effect of removing drift if true
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

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, sharex=True)
        ax1.set_title("x direction")
        ax1.plot(array_check[step_start:step_stop, 0], label="Before ")
        ax1.plot(array[step_start:step_stop, 0], label="After ")
        ax1.legend(loc='upper left')

        ax2.set_title("y direction")
        ax2.plot(array_check[step_start:step_stop, 1], label="Before")
        ax2.plot(array[step_start:step_stop, 1], label="After ")
        ax2.legend(loc='upper left')

        ax3.set_title(" z direction")
        ax3.plot(array_check[step_start:step_stop, 2], label="Before ")
        ax3.plot(array[step_start:step_stop, 2], label="After ")
        ax3.legend(loc='upper left')

        fig.text(0.5, 0.04, 'Sample number', ha='center')
        fig.text(0.04, 0.5, 'Velocity [m/s]', va='center', rotation='vertical')
        plt.show()

        plt.show()
    # endregion

    return array


def find_position(vel, touchdowns, fs):
    """
    Integrate velocity to get position. Removes drift after integration
    :param vel: velocity [m/s]
    :param touchdowns: np.array with 0 where leg is in motion and 1 where leg is at the ground
    :param fs: sampling frequency [Hz]
    :return: pos: position [m]
    """
    pos = np.zeros((len(vel), 3))
    for i in range(1, len(pos)):
        pos[i, :] = pos[i - 1, :] + vel[i, :] * 1 / fs

        if touchdowns[i] != 0:
            # pos[i, 0] = 0
            pos[i, :] = [0, 0, 0]

    # Remove drift
    pos = remove_drift(pos, touchdowns)

    return pos


# region Shortcut to what is done in NodeAlgorithm
if __name__ == '__main__':

    # Load df made in current_df.py
    df = pd.read_pickle('current_df.pkl')

    acc = np.asarray(df[['accX[mg]', 'accY[mg]', 'accZ[mg]']], dtype=float) * 10 ** -3
    gyro = np.asarray(df[['gyroX[mdps]', 'gyroY[mdps]', 'gyroZ[mdps]']], dtype=float) * 10 ** -3
    mag = np.asarray(df[['magX[mG]', 'magY[mG]', 'magZ[mG]']], dtype=float) * 10 ** -3

    # Rotate to earth frame
    # Want to change rotation to class such that the rotation itself is done in rotation.py, and not here
    acc_earth = [1, 0, 0]
    acc_body = acc[0, :]

    rot = rotation_matrix_from_vectors(acc_body, acc_earth)
    for i in range(len(acc[:, 0])):
        acc[i, :] = rot.dot(acc[i, :])
        gyro[i, :] = rot.dot(gyro[i, :])
        mag[i, :] = rot.dot(mag[i, :])

    dt = 1 / 100
    x = integrate.cumtrapz(integrate.cumtrapz(acc[:, 0], dx=dt, initial=0), dx=dt, initial=0)
    y = integrate.cumtrapz(integrate.cumtrapz(acc[:, 1], dx=dt, initial=0), dx=dt, initial=0)
    z = integrate.cumtrapz(integrate.cumtrapz(acc[:, 2], dx=dt, initial=0), dx=dt, initial=0)

    # Plot 3D Trajectory
    fig3, ax = plt.subplots()
    fig3.suptitle('3D Trajectory of leg')
    ax = plt.axes(projection='3d')
    ax.plot3D(x, y, z, c='red', lw=5, label='leg trajectory')
    ax.set_xlabel('X position (m)')
    ax.set_ylabel('Y position (m)')
    ax.set_zlabel('Z position (m)')
    plt.show()
# endregion
