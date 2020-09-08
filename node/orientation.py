import numpy as np
import matplotlib.pyplot as plt


def find_orientation_gyro(gyro, fs):
    """
    Integrate gyroscope data to find euler angles.
    :param acc: gyroscope data [deg/s]
    :param fs: sampling frequency [Hz]
    :return: rot: rotation in euler angles. Roll, pitch and yaw for for x, y and z
    """
    # Rotation = [roll, pitch, yaw]
    rot = np.zeros((len(gyro), 3))

    # Integrate acceleration
    for i in range(1, len(gyro)):
        rot0 = np.asarray(gyro[i - 1, :])
        omega = np.asarray(gyro[i, :])
        rot[i, :] = rot0 + omega * 1 / fs

    return rot


def find_roll(acc):
    """
    Find euler angle roll (rotation about y-axis/north).
    :param acc: acceleration in dim (len,3)
    :return: roll [deg]
    """
    x = acc[:, 0]
    y = acc[:, 1]
    z = acc[:, 2]

    return 0


def find_pitch(acc):
    """
    Find euler angle pitch (rotation about z-axis/east).
    :param acc: acceleration in dim (len,3)
    :return: pitch [deg]
    """
    x = acc[:, 0]
    y = acc[:, 1]
    z = acc[:, 2]

    return 0


def find_yaw(roll, pitch, mag):
    x = mag[:, 0]
    y = mag[:, 1]
    z = mag[:, 2]

    return 0


def dist(a, b):
    """
    Find distance between a and b
    :param a: point a
    :param b: point b
    :return: distance
    """
    return np.sqrt(a ** 2 + b ** 2)
