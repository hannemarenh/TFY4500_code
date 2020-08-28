#region Import packages and functions
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from node.result import Result
from node.rotation import rotate_to_earth_frame
from node.touchdown import *
from node.velocity import *
from node.position import find_position
#endregion


class NodeAlgorithm:
    # Default feature columns (name of what the sensor measures)
    feature_columns = ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z", "mag_x", "mag_y", "mag_z"]
    acc_earth = np.empty([2, 2], dtype=float)
    gyro_earth = np.empty([2, 2], dtype=float)
    mag_earth = np.empty([2, 2], dtype=float)

    def __init__(self, file=None, df=None, freq=100):
        """
        Constructor for NodeAlgorithm
        :param file: sensor data (containing data for feature columns). Default is None
        :param freq: Sampling frequency. Default set to 100 Hz
        """
        self.freq = freq
        self.result = Result()

        if (file is None) and (df is None):
            raise Exception("Either 'file' or 'df' must be specified.'")
        elif df is None:
            try:
                df = pd.read_csv(file, error_bad_lines=False)
            except:
                raise Exception("Error opening df")

        result = self.calculate(df)

    def calculate(self, df):
        """
        Calculate characteristics of leg motion
        :param df: sensor data (containing feature columns) as dataframe
        :return: result
        """
        # Extract feature columns
        acc_body = np.asarray(df[['accX[mg]', 'accY[mg]', 'accZ[mg]']], dtype=float) * 10 ** -3                 #[g]
        gyro_body = np.asarray(df[['gyroX[mdps]', 'gyroY[mdps]', 'gyroZ[mdps]']], dtype=float) * 10 ** -3       #[dps]
        mag_body = np.asarray(df[['magX[mG]', 'magY[mG]', 'magZ[mG]']], dtype=float) * 10 ** -3                 #[gauss]

        # Rotate to earth frame
        self.acc_earth, self.gyro_earth, self.mag_earth = rotate_to_earth_frame(acc_body, gyro_body, mag_body, plot_rotation=False)

        # Transform acc data to have SI units
        self.g_to_SI()

        # Find touchdowns
        touchdowns = find_touchdowns_gyro(self.gyro_earth, self.freq)
        touchdowns_acc = find_touchdowns_acc(self.acc_earth, self.freq)

        # Find velocity
        velocity = find_velocity(self.acc_earth, touchdowns, self.freq, plot_drift=True)
        velocity_cumtrapz = find_velocity_cumtrapz(self.acc_earth, self.freq)

        # Find position
        position = find_position(velocity, touchdowns, self.freq)

        return self.result

    def g_to_SI(self):
        self.acc_earth *= 9.807     #[m/s/s]
        self.acc_earth -= [9.087, 0, 0]





