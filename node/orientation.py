import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from node.rotation import change_coordinate_system, rotate_to_earth_frame


def find_rotation_matrix(acc, gyro, mag, fs):
    """
    Function to find total rotation matrix by multiplying roll, pitch and yaw rotation matrices
    :param acc: acceleration data [m/s/s]
    :param gyro: gyroscope data [deg/s]
    :param mag: magnetometer data [gauss]
    :param fs: sampling frequency [Hz]
    :return: Rotation matrix Rxyz
    """
    euler_angles = find_orientation(acc, gyro, mag, fs)

    phi = euler_angles[:, 0]
    theta = euler_angles[:, 1]
    psi = euler_angles[:, 2]

    # Test vector with initial orientation pointing down along z axis.
    orientation = np.zeros((len(acc), 3))
    orientation[0, :] = [0, 0, 1]

    Rxyz = np.zeros((len(acc), 3, 3))

    for i in range(1, len(acc)):
        Rx = np.array([[1, 0, 0], [0, np.cos(phi[i]), np.sin(phi[i])], [0, -np.sin(phi[i]), np.cos(phi[i])]])
        Ry = np.array([[np.cos(theta[i]), 0, -np.sin(theta[i])], [0, 1, 0], [np.sin(theta[i]), 0, np.cos(theta[i])]])
        Rz = np.array([[np.cos(psi[i]), np.sin(psi[i]), 0], [-np.sin(psi[i]), np.cos(psi[i]), 0], [0, 0, 1]])

        Rxyz[i, :, :] = (Rx @ Ry) @ Rz
        orientation[i] = Rxyz[i, :, :].dot(orientation[i - 1])

    return Rxyz


def find_orientation_gyro(gyro, fs):
    """
    Integrate gyroscope data to find euler angles.
    :param gyro: gyroscope data [deg/s]
    :param fs: sampling frequency [Hz]
    :return: rot: rotation in euler angles [deg]. Phi (roll),theta (pitch) and psi (yaw) for for x, y and z
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
    :return: rot: rotation in euler angles [deg]. Phi (roll),theta (pitch) and psi (yaw) for for x, y and z
    """
    # Rotation = [roll, pitch, yaw]
    rot = np.zeros((len(acc), 3))

    rot[:, 0] = find_roll(acc)
    rot[:, 1] = find_pitch(acc)
    rot[:, 2] = find_yaw(mag)

    return rot


def find_orientation(acc, gyro, mag, fs):
    """
    Find orientation using complimentary filter
    :param acc: acceleration data [m/s/s]
    :param gyro: gyroscope data [deg/s]
    :param mag: magnetometer data [gauss]
    :param fs: sampling frequency [Hz]
    :return: rot: rotation in euler angles [deg]. Phi (roll),theta (pitch) and psi (yaw) for for x, y and z
    """
    # Rotation = [roll, pitch, yaw]
    rot = np.zeros((len(acc), 3))

    # Constants for filter
    K1 = 0.98
    K2 = 1 - K1

    rot_gyro = find_orientation_gyro(gyro, fs)
    rot_acc_mag = find_orientation_acc_mag(acc, mag)

    rot[:, 0] = K1 * rot_gyro[:, 0] + K2 * rot_acc_mag[:, 0]
    rot[:, 1] = K1 * rot_gyro[:, 1] + K2 * rot_acc_mag[:, 1]
    rot[:, 2] = K1 * rot_gyro[:, 2] + K2 * rot_acc_mag[:, 2]

    return rot


def find_roll(acc):
    """
    Find euler angle phi (roll rotation about x-axis/north).
    :param acc: acceleration in dim (len,3) [m/s/s]
    :return: phi [deg]
    """
    x = acc[:, 0]
    y = acc[:, 1]
    z = acc[:, 2]

    phi = np.arctan2(y, x) * 180 / np.pi

    # Make rotation matrix for checking while debugging
    Rx = np.zeros((len(x), 3, 3))
    for i in range(1, len(acc)):
        Rx[i, :, :] = np.array([[1, 0, 0], [0, np.cos(phi[i]), np.sin(phi[i])], [0, -np.sin(phi[i]), np.cos(phi[i])]])

    return phi


def find_pitch(acc):
    """
    Find euler angle theta (pitch rotation about y-axis/east).
    :param acc: acceleration in dim (len,3) [m/s/s]
    :return: theta [deg]
    """
    x = acc[:, 0]
    y = acc[:, 1]
    z = acc[:, 2]

    theta = np.arctan(-x / dist(y, z)) * 180 / np.pi

    # Make rotation matrix for checking while debugging
    Ry = np.zeros((len(x), 3, 3))
    for i in range(1, len(x)):
        Ry[i, :, :] = np.array(
            [[np.cos(theta[i]), 0, -np.sin(theta[i])], [0, 1, 0], [np.sin(theta[i]), 0, np.cos(theta[i])]])

    return theta


def find_yaw(mag):
    """
    Find euler angle psi (yaw rotation arount z-axis/down)
    :param mag: magnetic field in dim (len,3)
    :return: psi (deg)
    """
    x = mag[:, 0]
    y = mag[:, 1]
    z = mag[:, 2]

    psi = np.arctan2(y, x) * 180 / np.pi

    # Make rotation matrix for checking while debugging
    Rz = np.zeros((len(x), 3, 3))
    for i in range(1, len(x)):
        Rz[i, :, :] = np.array([[np.cos(psi[i]), np.sin(psi[i]), 0], [-np.sin(psi[i]), np.cos(psi[i]), 0], [0, 0, 1]])

    return psi


def dist(a, b):
    """
    Find distance between a and b
    :param a: point a
    :param b: point b
    :return: distance
    """
    return np.sqrt(a ** 2 + b ** 2)


if __name__ == '__main__':
    # Load and prepare csv file to dataframe
    # Load csv file. Skip lines where measurements are missing

    # Three datasets with easy rotations for checking code.
    # Initial pos like on left horse leg
    # Rotating approx. 45 deg around given axis (coord system used in code, not in sensor [z down, x forward])
    title0 = r"x_roll45.csv"
    title1 = r"y_pitch45.csv"
    title2 = r"z_yaw45.csv"

    file = r"C:\\Users\\Hanne Maren\\Documents\\Prosjektoppgave\\Data\\control\\" + title0
    df = pd.read_csv(file, error_bad_lines=False)

    # Remove "forskyvede" rows
    # Dont know why they occur, but they do... :( Probably something with the sensor
    # For sensorTile
    check = df.iloc[:, -1].notnull()
    for i in range(0, len(check)):
        if check[i]:
            df = df.drop(i)

    # Delete nan rows
    df = df.iloc[:, : -1].dropna(axis=0)

    freq = 100  # [Hz]

    # Extract feature columns and change to new coordinate system, where z is up, y is out and x is forward
    # when sensor is placed with USB in and screws out on left foot
    acc_body = change_coordinate_system(
        np.asarray(df[['accX[mg]', 'accY[mg]', 'accZ[mg]']], dtype=float) * 10 ** -3)  # [g]
    gyro_body = change_coordinate_system(
        np.asarray(df[['gyroX[mdps]', 'gyroY[mdps]', 'gyroZ[mdps]']], dtype=float) * 10 ** -3)  # [dps]
    mag_body = change_coordinate_system(
        np.asarray(df[['magX[mG]', 'magY[mG]', 'magZ[mG]']], dtype=float) * 10 ** -3)  # [gauss]

    # Rotate to earth frame of reference
    acc_earth, gyro_earth, mag_earth = rotate_to_earth_frame(acc_body, gyro_body, mag_body,
                                                             freq, plot_rotation=False)
    # Transform acc data to have SI units
    # acc_earth *= 9.807  # [m/s/s]
    # acc_earth -= [0, 0, 9.807]

    # Find rotation matrix Rxyz
    # At this point, a orientation vector is also made for checking inside the function
    find_rotation_matrix(acc_earth, gyro_earth, mag_earth, freq)
