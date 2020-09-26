import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from node.rotation import change_coordinate_system, rotate_to_earth_frame
from node.touchdown import filterHP, filterLP
from node.animation import plot_orientation, plot_position
from node.velocity import remove_drift
from node.gait_tracking import ahrs
from node.gait_tracking.quaternion_utils import *


def find_orientation(acc, gyro, mag, touchdowns, fs, plot_drift=False):
    """
    Find orientation using complimentary filter
    :param acc: acceleration data [m/s/s]
    :param gyro: gyroscope data [deg/s]
    :param mag: magnetometer data [gauss]
    :param fs: sampling frequency [Hz]
    :return: acc_oriented: Acceleration data with orientation calculated by complimentary filter
             x_orientation: Vector showing orientation for x-axis
             y_orientation: Vector showing orientation for y-axis
             z_orientation: Vector showing orientation for z-axis
    """
    # Rotation = [roll, pitch, yaw]
    angles = np.zeros((len(acc), 3))

    # Constants for filter
    K1 = 0.98
    K2 = 1 - K1

    quat_acc_gyro_magn, quat_acc_gyro = find_quaternions(acc, gyro, mag, fs, touchdowns)
    R = quaternion_to_rotmat(quat_acc_gyro_magn)

    angles_gyro = find_angles_gyro(gyro, touchdowns, fs, plot_drift)
    angles_acc_mag = find_angles_acc_mag(acc, mag, fs)

    # Complimentary filter
    # Rotation from acceleration and mag does not make sense.... Use gyro rotation
    #angles[:, 0] = K1 * angles_gyro[:, 0] + K2 * angles_acc_mag[:, 0]
    #angles[:, 1] = K1 * angles_gyro[:, 1] + K2 * angles_acc_mag[:, 1]
    #angles[:, 2] = K1 * angles_gyro[:, 2] + K2 * angles_acc_mag[:, 2]

    angles = angles_gyro                                                         # remove this one if euler angles from acc+mag suddenlty works
    dangles = np.concatenate((np.array([[0, 0, 0]]), np.diff(angles, axis=0)))   # Find how much angles changes from ponit to point

    # Initialize orientation vectors
    x_orientation, y_orientation, z_orientation = np.zeros((len(angles), 3)), np.zeros((len(angles), 3)), np.zeros((len(angles), 3))

    # Magnitude of orientation vectors
    m = 1
    x_orientation[0, :] = [m, 0, 0]
    y_orientation[0, :] = [0, m, 0]
    z_orientation[0, :] = [0, 0, m]

    acc_oriented = np.zeros((len(acc), 3))
    acc_oriented[0, :] = R[:, :, 0].dot(acc[0, :])
    for i in range(1, len(acc)):
        '''
        Euler angles
        # Find rotation matrices for the angle
        Rx = rotmat_x(dangles[i, 0])
        Ry = rotmat_y(dangles[i, 1])
        Rz = rotmat_z(dangles[i, 2])

        # Make total rotation matrix
        Rzyx = (Rz @ Ry) @ Rx

        # Rotate acceleration data
        acc_oriented[i, :] = Rzyx.dot(acc[i-1, :])
        

        # Rotate orientation vectors
        x_orientation[i, :] = Rzyx.dot(x_orientation[i - 1, :])
        y_orientation[i, :] = Rzyx.dot(y_orientation[i - 1, :])
        z_orientation[i, :] = Rzyx.dot(z_orientation[i - 1, :])
        '''
        #Quaternions
        if i==225:
            plot_orientation(x_orientation[0:225 ,:], y_orientation[0:225 ,:], z_orientation[0:225 ,:])
        acc_oriented[i, :] = R[:, :, i].dot(acc[0, :])
        x_orientation[i, :] = R[:, :, i].dot(x_orientation[0, :])
        y_orientation[i, :] = R[:, :, i].dot(y_orientation[0, :])
        z_orientation[i, :] = R[:, :, i].dot(z_orientation[0, :])

    """
    # Code for checking orientation (using code from this file!!)
    plt.plot(angles[:, 0], label="x rotation, roll (phi)")
    plt.plot(angles[:, 1], label="y rotation, pitch (theta)")
    plt.plot(angles[:, 2], label="z rotation, yaw (psi)")
    plt.ylabel('Rotation [deg]')
    plt.xlabel('Sample')
    plt.legend()
    plt.show()

    start = 225
    end = 300
    plot_orientation(x_orientation[start:end, :], y_orientation[start:end, :], z_orientation[start:end, :])
    """

    return acc_oriented, x_orientation, y_orientation, z_orientation


def find_quaternions(acc, gyro, mag, fs, touchdowns=None):
    # Computed using AHRS (attitude and heading reference system) algorithm
    initial_orientation = [1, 0, 0, 0]

    quat_acc_gyro_mag = []
    quat_acc_gyro = []
    alg_acc_gyro_mag = ahrs.AHRS(sample_period=1 / fs, kp=1, kp_init=1, initial_orientation=initial_orientation)
    alg_acc_gyro = ahrs.AHRS(sample_period=1 / fs, kp=1, kp_init=1, initial_orientation=initial_orientation)

    for i in np.arange(len(acc)):
        if touchdowns is None:
            alg_acc_gyro_mag.kp = 0
            alg_acc_gyro.kp = 0
        elif touchdowns[i]:
            alg_acc_gyro_mag.kp = 0.5
            alg_acc_gyro.kp = 0.5

        alg_acc_gyro_mag.update(gyro[i, :] * np.pi / 180, acc[i, :], mag[i, :])
        alg_acc_gyro.update_imu(gyro[i, :] * np.pi / 180, acc[i, :])

        quat_acc_gyro_mag.append(alg_acc_gyro_mag.quaternion())
        quat_acc_gyro.append(alg_acc_gyro.quaternion())

    return np.array(quat_acc_gyro_mag), np.array(quat_acc_gyro)


def find_angles_gyro(gyro, touchdowns, fs, plot_drift=False):
    """
    Integrate gyroscope data to find rotation angles for each axis.
    :param gyro: gyroscope data [deg/s]
    :param fs: sampling frequency [Hz]
    :return: angles: low pass filtered rotation angles around each axis[deg]. Phi (roll),theta (pitch) and psi (yaw) for for x, y and z
    """
    # Rotation = [roll, pitch, yaw]
    angles = np.zeros((len(gyro), 3))

    # Integrate acceleration
    for i in range(1, len(gyro)):
        # Find current angle in degrees
        angle0 = np.asarray(angles[i - 1, :])
        omega = np.asarray(gyro[i, :])
        angles[i, :] = angle0 + omega * 1 / fs

        # Force rotation to be zero when foot is touching ground
        if touchdowns[i] != 0:
            angles[i, :] = [0, 0, 0]

    # Remove drift
    angles = remove_drift(angles, touchdowns, fs, plot_drift, 'Angle [deg]')

    lp_angles = filterLP(3, 3, fs, angles)
    return lp_angles


def find_angles_acc_mag(acc, mag, fs):
    """
    Use accelerometer and magnetometer to find euler angles.
    :param acc: acceleration data [m/s/s]
    :param mag: magnetometer data [gauss]
    :return: rot: rotation in euler angles [deg]. Phi (roll),theta (pitch) and psi (yaw) for for x, y and z
    """
    # Rotation = [roll, pitch, yaw]
    angles = np.zeros((len(acc), 3))

    angles[:, 0] = find_roll(acc)
    angles[:, 1] = find_pitch(acc)
    angles[:, 2] = find_yaw(mag)

    lp_angles = filterLP(3, 3, fs, angles)
    return lp_angles


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


def rotmat_x(degrees):
    """
    Function that generates a rotation matrix around x axis for a given angle given in degrees.
    :param degrees: angle [degrees]
    :return: Rotation matrix around x axis
    """
    rad = np.deg2rad(degrees)
    return np.array([[1, 0, 0], [0, np.cos(rad), -np.sin(rad)], [0, np.sin(rad), np.cos(rad)]])


def rotmat_y(degrees):
    """
    Function that generates a rotation matrix around y axis for a given angle given in degrees.
    :param degrees: angle [degrees]
    :return: Rotation matrix around y axis
    """
    rad = np.deg2rad(degrees)
    return np.array([[np.cos(rad), 0, np.sin(rad)], [0, 1, 0], [-np.sin(rad), 0, np.cos(rad)]])


def rotmat_z(degrees):
    """
    Function that generates a rotation matrix around z axis for a given angle given in degrees.
    :param degrees: angle [degrees]
    :return: Rotation matrix around z axis
    """
    rad = np.deg2rad(degrees)
    return np.array([[np.cos(rad), -np.sin(rad), 0], [np.sin(rad), np.cos(rad), 0], [0, 0, 1]])


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

    # Extract feature columns and change to new coordinate system, where z is down, y is in and x is forward
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
    acc_earth -= [0, 0, 1]

    # Find rotation matrix Rxyz
    # At this point, a orientation vector is also made for checking inside the function
    lp_acc_earth = filterLP(1, 5, freq, acc_earth)
    lp_gyro_earth = filterLP(1, 5, freq, gyro_earth)
    lp_mag_earth = filterLP(1, 5, freq, mag_earth)

    touchdowns = np.zeros(len(acc_earth))
    acc_new, x_ori, y_ori, z_ori = find_orientation(acc_earth, gyro_earth, mag_earth, touchdowns, freq, plot_drift=True)

    """
plt.plot(angles[:,0],  label="x rotation, roll (phi)")
plt.plot(angles[:,1],  label="y rotation, pitch (theta)")
plt.plot(angles[:,2],  label="z rotation, yaw (psi)")
plt.legend()
plt.title('Pitch check')
"""
