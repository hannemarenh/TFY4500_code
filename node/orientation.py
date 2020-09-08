import numpy as np
import matplotlib.pyplot as plt


def find_orientation_gyro(gyro, fs):
    """
    Integrate gyroscope data to find euler angles.
    :param acc: acceleration data [m/s/s]
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


def find_orientation_acc_mag(acc, mag):
    """
    Use accelerometer and magnetometer to find euler angles.
    :param acc: acceleration data [m/s/s]
    :param mag: magnetometer data [gauss]
    :return: rot: rotation in euler angles. Roll, pitch and yaw for for x, y and z
    """
    # Rotation = [roll, pitch, yaw]
    rot = np.zeros((len(acc), 3))

    rot[:, 0] = find_roll(acc)
    rot[:, 1] = find_pitch(acc)
    rot[:, 2] = find_yaw(mag)

    return rot


def find_orientation(acc, gyro, mag, fs):
    """
    FInd orientation using complimentary filter
    :param acc: acceleration data [m/s/s]
    :param gyro: gyroscope data [deg/s]
    :param mag: magnetometer data [gauss]
    :param fs: sampling frequency [Hz]
    :return: rot: rotation in euler angles. Roll, pitch and yaw for for x, y and z
    """
    # Rotation = [roll, pitch, yaw]
    rot = np.zeros((len(acc), 3))

    #Constants for filter
    K1 = 0.98
    K2 = 1-K1

    rot_gyro = find_orientation_gyro(gyro, fs)
    rot_acc_mag = find_orientation_acc_mag(acc, mag)

    rot[:, 0] = K1 * rot_gyro[:, 0] + K2 * rot_acc_mag[:, 0]
    rot[:, 1] = K1 * rot_gyro[:, 1] + K2 * rot_acc_mag[:, 1]
    rot[:, 2] = K1 * rot_gyro[:, 2] + K2 * rot_acc_mag[:, 2]

    return  rot


def find_roll(acc):
    """
    Find euler angle phi (roll rotation about x-axis/north).
    :param acc: acceleration in dim (len,3)
    :return: roll [deg]
    """
    x = acc[:, 0]
    y = acc[:, 1]
    z = acc[:, 2]

    phi = np.arctan2(y, x) * 180/np.pi
    # phi = np.arc(y, dist(x,z)) *180 / np.pi           #Not sure which one to use
    return phi


def find_pitch(acc):
    """
    Find euler angle theta (pitch rotation about y-axis/east).
    :param acc: acceleration in dim (len,3)
    :return: pitch [deg]
    """
    x = acc[:, 0]
    y = acc[:, 1]
    z = acc[:, 2]

    theta = np.arctan(-x/dist(y, z)) * 180/np.pi
    return theta


def find_yaw(mag):
    """
    Find euler angle psi (yaw rotation arount z-axis/down)
    :param mag: magnetic field in dim (len,3)
    :return: yaw (deg)
    """
    x = mag[:, 0]
    y = mag[:, 1]
    z = mag[:, 2]

    yaw = np.arctan2(y, x) * 180/np.pi
    return 0


def dist(a, b):
    """
    Find distance between a and b
    :param a: point a
    :param b: point b
    :return: distance
    """
    return np.sqrt(a ** 2 + b ** 2)
