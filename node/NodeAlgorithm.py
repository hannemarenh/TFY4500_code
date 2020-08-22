#region Import packages and functions
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from node.result import Result

#endregion


class NodeAlgorithm:
    # Default feature columns (name of what the sensor measures)
    feature_columns = ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z", "mag_x", "mag_y", "mag_z"]

    def __init__(self, file=None, freq=100):
        """
        Constructor for NodeAlgorithm
        :param file: sensor data (containing data for feature columns). Default is None
        :param freq: Sampling frequency. Default set to 100 Hz
        """
        self.freq = freq
        self.result = Result()

        if file is None:
            raise Exception("'file' must be specified.'")
        else:
            try:
                df = pd.read_csv(file, error_bad_lines=False)
            except:
                raise Exception("Error opening df")

            result = self.calculate(df)

    def calculate(self, df):
        """
        Calculate characteristics of leg motion
        :param df: sensor data (containing feature columns) as dataframe
        :return:
        """
        # Isolate the measured data in separate variables
        acc, gyro, mag = self.prepare(df)


        return self.result

    def prepare(self, df):
        '''
        Prepare the data by converting to SI units and make numpy arrays for each of the measured parameters.
        OBS. Acc is converted after begind rotated in the rotate() function
        :param df: pandas dataframe iwth the measured parameters
        :return: numpy arrays for acc, gyro and mag in SI units. Acc is rotated
        '''
        # Isolate acceleration data
        acc = np.asarray(df[['accX[mg]', 'accY[mg]', 'accZ[mg]']], dtype=float) * 10 ** -3   #[g]

        # Rotate acc to new frame of reference (x perfect down, y perfect back/forth, z perfect in/out)
        acc_check = acc.copy()*9.81
        acc = self.rotate(acc)

        # Isolate gyroscope data
        deg2rad = 2 * np.pi / 360
        gyro = np.asarray(df[['gyroX[mdps]', 'gyroY[mdps]', 'gyroZ[mdps]']]) * 10 ** -3 * deg2rad  # [rad/s]

        # Isolate magnetometer data
        mag = np.asarray(df[['magX[mG]', 'magY[mG]', 'magZ[mG]']]) * 10**-3                   #[G, 1G = 10**-4T]

        return acc, gyro, mag

    @staticmethod
    def rotate(acc):
        """
        Rotatiing acc data to new frame of reference. Using first 5 seconds to calibrate
        :param acc: acceleration in [g] before rotation
        :return: acc in [m/s/s] after rotation
        """
        # The wanted acc
        acc_new = np.array([1., 0., 0.])  # [g]

        #Make copy of acc just for controlling the rotation
        acc_check = acc.copy()

        # normalize
        acc_old = (np.mean(acc[:500, :], axis=0) / np.linalg.norm(np.mean(acc[:500, :], axis=0)))

        # ref: https://math.stackexchange.com/questions/180418/calculate-rotation-matrix-to-align-vector-a-to-vector-b-in-3d/476311#476311
        cross = np.cross(acc_old, acc_new)
        dot = np.dot(acc_old, acc_new)
        cross_norm = np.linalg.norm(cross)
        acc_old_norm = np.linalg.norm(acc_old)
        identity = np.eye(3)
        vX = np.array([[0, -cross[2], cross[1]], [cross[2], 0, -cross[0]], [-cross[1], cross[0], 0]])

        r = (identity + vX + np.matmul(vX, vX) * ((1 - dot) / (cross_norm ** 2))) / acc_old_norm ** 2

        # Rotate accereration data
        for i in range(len(acc)):
            acc[i] = r.dot(acc[i])

        # Remove gravity
        acc[:, 0] = acc[:, 0] - acc_new[0]

        #region Control plots
        plt.figure()
        plt.plot(acc[:, 0], 'g', label='rotated acc_x')
        plt.plot(acc_check[:, 0], 'r', label='non rotated acc_x')
        plt.legend()
        plt.show()

        plt.figure()
        plt.plot(acc[:, 1], 'g', label='rotated acc_y')
        plt.plot(acc_check[:, 1], 'r', label='non rotated acc_y')
        plt.legend()
        plt.show()

        plt.figure()
        plt.plot(acc[:, 2], 'g', label='rotated acc_z')
        plt.plot(acc_check[:, 2], 'r', label='non rotated acc_z')
        plt.legend()
        plt.show()
        #endregion

        return acc*9.81

# Testing. Later moved to run_node
title = r"hm200720_walk.csv"
file = r"C:\Users\Hanne Maren\Documents\nivo\project\legData\\" + title
leg = NodeAlgorithm(file)



