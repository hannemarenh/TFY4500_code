import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from node.rotation import change_coordinate_system, rotate_to_earth_frame
from node.touchdown import filterHP, filterLP
from node.gait_tracking.quaternion_utils import *
from node.gait_tracking import ahrs
import scipy


def find_acc_lab_quat(acc_body, gyro_body, mag_body, fs, touchdowns=None):
    # Computed using AHRS (attitude and heading reference system) algorithm
    initial_orientation = [1, 0, 0, 0]

    quat_acc_gyro_mag = []
    quat_acc_gyro = []
    alg_acc_gyro_mag = ahrs.AHRS(sample_period=1 / fs, kp=1, kp_init=1, initial_orientation=initial_orientation)
    alg_acc_gyro = ahrs.AHRS(sample_period=1 / fs, kp=1, kp_init=1, initial_orientation=initial_orientation)

    for i in np.arange(len(acc_body)):
        if touchdowns is None:
            alg_acc_gyro_mag.kp = 0
            alg_acc_gyro.kp = 0
        elif touchdowns[i]:
            alg_acc_gyro_mag.kp = 0.5
            alg_acc_gyro.kp = 0.5

        alg_acc_gyro_mag.update(gyro_body[i, :] * np.pi / 180, acc_body[i, :], mag_body[i, :])
        alg_acc_gyro.update_imu(gyro_body[i, :] * np.pi / 180, acc_body[i, :])

        quat_acc_gyro_mag.append(alg_acc_gyro_mag.quaternion())
        quat_acc_gyro.append(alg_acc_gyro.quaternion())

    # Choose either with or without mag data in calculation of quaternions
    quat = np.array(quat_acc_gyro_mag)
    R = quaternion_to_rotmat(quat)

    # Calculate acc_lab by rotation acc_body
    acc_lab = np.zeros((len(acc_body), 3))
    for i in range(len(acc_body)):
        acc_lab[i, :] = R[:, :, i].dot(acc_body[0, :])

    return acc_lab


def find_acc_lab_euler(acc_body, gyro_body, fs):
    '''
    Function to rotate acceleration data from body frame to fixed lab frame using euler angles
    :param acc_body: Measured acceleration with dim (len, 3)
    :param gyro_body: Measured angulat velocity [deg/s] with dim (len, 3)
    :param fs: sampling freq [Hz]
    :return: vacc_lab, acceleration in lab frame
    '''
    # Change to radians per second
    gyro_body = gyro_body * np.pi / 180

    # Euler angles [phi, theta, roll]
    angles = np.zeros((len(acc_body), 3))

    # Derivative of Euler angles [dphi, dtheta, droll]
    dangles = np.zeros((len(acc_body), 3))

    # Inverse rotation matrix for gyroscope data
    M_inv = getM_inv(angles[0, 1]+0.01, angles[0, 2]+0.01)

    # Inverse rotation matrix for acceleration data
    R_inv = getR_inv(angles[0, 0], angles[0, 1], angles[0, 2])

    # Initialize acceleration array in lab frame
    acc_lab = np.zeros((len(acc_body), 3))
    acc_lab[0, :] = R_inv @ acc_body[0, :]

    # Time step
    dt = 1/fs

    for i in range(1, len(acc_body)):
        # Rotate dangles to lab_frame
        dangles[i, :] = M_inv @ gyro_body[i, :]

        # Integrate dangles
        angles[i, :] = angles[i-1, :] + dangles[i, :] * dt

        # Update rotation matrix R
        R_inv = getR_inv(angles[i, 0], angles[i, 1], angles[i, 2])

        # Rotate from acc_body to acc_lab
        acc_lab[i, :] = R_inv @ acc_body[i, :]

        # Update rotation matrix M
        M_inv = getM_inv(angles[i, 1], angles[i, 2])

    return acc_lab


def getM_inv(theta, psi):
    '''
    Rotation matrix for gyroscope data is M, g_body = M g_lab. This function gives the inverse rotation matrix, to get
    g_lab from g_body, ie. g_lab = M_inv g_body
    :param theta: Rotation about x'' axis in radians
    :param psi: Rotation about z'' ' axis in radians
    :return: M_inv, inverse rotation matrix
    '''
    # Calculate M
    M = np.array([[np.sin(theta)*np.sin(psi), np.cos(psi), 0],
                     [np.sin(theta)*np.cos(psi), -np.sin(psi), 0],
                     [np.cos(theta), 0, 1]])

    # If matrix is singular a small angle is added
    if np.linalg.det(M) == 0:
        if theta == 0:
            return getM_inv(theta + 0.001, psi)
        elif psi == 0:
            return getM_inv(theta, psi + 0.001)
        else:
            return getM_inv(theta + 0.01, psi + 0.001)

    # Return inverse M
    return np.linalg.inv(M)


def getR_inv(phi, theta, psi):
    '''
    Get rotation matrix for lab to body frame, a_body = R a_lab. This function gives the inverse rotation matrix, to get
    a_lab from a_body, ie. a_lab = R_inv a_body
    :param phi: Rotation about z' axis in radians
    :param theta: Rotation about x'' axis in radians
    :param psi: Rotation about z'' ' axis in radians
    :return: R_inv, inverse rotation matrix
    '''
    R_phi = np.array([[np.cos(phi), np.sin(phi), 0],
                     [-np.sin(phi), np.cos(phi), 0],
                     [0, 0, 1]])

    R_theta = np.array([[1, 0, 0],
                     [0, np.cos(theta), np.sin(theta)],
                     [0, -np.sin(theta), np.cos(theta)]])

    R_psi = np.array([[np.cos(psi), np.sin(psi), 0],
                     [-np.sin(psi), np.cos(psi), 0],
                     [0, 0, 1]])

    R = R_psi @ (R_theta @ R_phi)

    return np.linalg.inv(R)


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

    fs = 100  # [Hz]

    #acc_body = np.asarray(df[['accX[mg]', 'accY[mg]', 'accZ[mg]']], dtype=float) * 10 ** -3              # [g]
    #gyro_body = np.asarray(df[['gyroX[mdps]', 'gyroY[mdps]', 'gyroZ[mdps]']], dtype=float) * 10 ** -3    # [dps]
    #mag_body = np.asarray(df[['magX[mG]', 'magY[mG]', 'magZ[mG]']], dtype=float) * 10 ** -3              # [gauss]

    acc_body = change_coordinate_system(
        np.asarray(df[['accX[mg]', 'accY[mg]', 'accZ[mg]']], dtype=float) * 10 ** -3)  # [g]
    gyro_body = change_coordinate_system(
        np.asarray(df[['gyroX[mdps]', 'gyroY[mdps]', 'gyroZ[mdps]']], dtype=float) * 10 ** -3)  # [dps]
    mag_body = change_coordinate_system(
        np.asarray(df[['magX[mG]', 'magY[mG]', 'magZ[mG]']], dtype=float) * 10 ** -3)  # [gauss]

    # Positive rotation counterclockwise
    gyro_body[:, 1] *= -1
    gyro_body[:, 2] *= -1

    lp_acc_body = filterLP(1, 5, fs, acc_body)
    lp_gyro_body = filterLP(1, 5, fs, gyro_body)

    bias = np.mean(lp_acc_body[:fs * 1, :], axis=0)

    lp_acc_body = lp_acc_body - bias


    #get_acc_lab(acc_body, gyro_body, fs)
    find_acc_lab_euler(lp_acc_body, lp_gyro_body, fs)