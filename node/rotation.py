import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def check_rotation(acc_earth, acc_body, fs):
    """
    Compare the two frames (body and earth).
    :param acc_earth: Acceleration [g] in earth frame
    :param acc_body: Acceleration [g] in body frame
    :return: 0
    """
    time = np.linspace(0, len(acc_earth)/fs, len(acc_earth))

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True)
    ax1.set_title("x direction")
    # ax1.set_ylabel("Axxeleration [g]")
    ax1.plot(time, acc_body[:, 0], 'r', label="Body frame")
    ax1.plot(time, acc_earth[:, 0], 'g:', label="Earth frame")
    ax1.legend(loc=1)

    ax2.set_title("y direction")
    ax2.plot(time, acc_body[:, 1], 'r', label="Body frame")
    ax2.plot(time, acc_earth[:, 1], 'g:', label="Earth frame")
    ax2.legend(loc=1)

    ax3.set_title("z direction")
    ax3.plot(time, acc_body[:, 2], 'r', label="Body frame")
    ax3.plot(time, acc_earth[:, 2], 'g:', label="Earth frame")
    ax3.legend(loc=1)

    fig.text(0.5, 0.04, 'Time [s]', ha='center')
    fig.text(0.04, 0.5, 'Acceleration [g]', va='center', rotation='vertical')
    plt.show()
    return 0


def rotation_matrix_from_vectors(vec1, vec2):
    """ Find the rotation matrix that aligns vec1 to vec2
    :param vec1: A 3d "source" vector
    :param vec2: A 3d "destination" vector
    :return mat: A transform matrix (3x3) which when applied to vec1, aligns it with vec2.
    """
    a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
    return rotation_matrix


def rotate_to_earth_frame(acc_body, gyro_body, mag_body, fs=100, plot_rotation=False):
    """
    Rotates from body frame of reference to earth frame of reference using rotation matrix.
    :param acc_body: np.array with shape (len,3) with acceleration data in body frame
    :param gyro_body: np.array with shape (len,3) with gyroscope data in body frame
    :param mag_body: np.array with shape (len,3) with magnetometer data in body frame
    :param fs: sampling frequency [Hz]
    :param plot_rotation: Plots acc before and after rotation
    :return: acc, gyro and mag in earth frame
    """
    acc_earth_calib = [0, 0, -1]
    acc_body_calib = acc_body[0, :]

    rot = rotation_matrix_from_vectors(acc_body_calib, acc_earth_calib)

    # Initialize arrays for earth frame
    acc_earth = acc_body.copy()
    gyro_earth = gyro_body.copy()
    mag_earth = mag_body.copy()

    n = len(acc_body[:, 0])
    for i in range(n):
        acc_earth[i, :] = rot.dot(acc_body[i, :])
        gyro_earth[i, :] = rot.dot(gyro_body[i, :])
        mag_earth[i, :] = rot.dot(mag_body[i, :])

    if plot_rotation:
        check_rotation(acc_earth, acc_body, fs)

    return acc_earth, gyro_earth, mag_earth


def change_coordinate_system(array):
    """
    Change to new coordinate system, where z is up, y is out and x is forward
    when sensor is placed with USB in and screws out on right foot

    Old coordinate system is such that x is down, y is backwards and z is out
    for the same sensor location
    :param array: array with dim (len,3) in the old coordinate system
    :return: new_array: array with dim(len,3) in the new coordinate system
    """
    new_array = array.copy()
    new_array[:, 0] = -array[:, 1]      # -x --> x
    new_array[:, 1] = array[:, 2]       # z --> y
    new_array[:, 2] = -array[:, 0]      # -x --> z

    return new_array
