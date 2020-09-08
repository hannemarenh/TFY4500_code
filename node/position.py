# region import packages
import numpy as np
import pandas as pd
from node.rotation import rotation_matrix_from_vectors
from scipy import integrate
import matplotlib.pyplot as plt
from scipy.integrate import cumtrapz
from scipy.fft import fft


# endregion


def find_position(vel, touchdowns, fs, plot_drift=False):
    """
    Integrate velocity to get position. Removes drift after integration
    :param vel: velocity [m/s]
    :param touchdowns: np.array with 0 where leg is in motion and 1 where leg is at the ground
    :param fs: sampling frequency [Hz]
    :param plot_drift: Plots effect of removing drift if true

    :return: pos: position [m]
    """
    pos = np.zeros((len(vel), 3))
    for i in range(1, len(pos)):
        pos[i, :] = pos[i - 1, :] + vel[i, :] * 1 / fs

        #Cannot set to zero... Think maybe drift removal at velocity cal is enough
        #if touchdowns[i] != 0:
            #pos[i, 0] = 0
            #pos[i, :] = [0, 0, 0]
    return pos


def find_position_cumtrapz(vel, touchdowns, fs, plot_drift=False):
    pos = np.zeros(vel.shape)
    pos[:, 0] = cumtrapz(vel[:, 0], dx=1 / fs, initial=0)
    pos[:, 1] = cumtrapz(vel[:, 1], dx=1 / fs, initial=0)
    pos[:, 2] = cumtrapz(vel[:, 2], dx=1 / fs, initial=0)

    return pos


# region Shortcut to what is done in NodeAlgorithm. Only used in first phase of debugging.
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
