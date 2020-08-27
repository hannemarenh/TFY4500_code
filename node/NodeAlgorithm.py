#region Import packages and functions
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from node.result import Result
from node.rotation import rotate_to_earth_frame
from node.touchdown import find_touchdowns
from node.velocity import find_velocity
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
        :return:
        """
        # Rotate from body frame to earth frame
        acc_body = np.asarray(df[['accX[mg]', 'accY[mg]', 'accZ[mg]']], dtype=float) * 10 ** -3                 #[g]
        gyro_body = np.asarray(df[['gyroX[mdps]', 'gyroY[mdps]', 'gyroZ[mdps]']], dtype=float) * 10 ** -3       #[dps]
        mag_body = np.asarray(df[['magX[mG]', 'magY[mG]', 'magZ[mG]']], dtype=float) * 10 ** -3                 #[gauss]

        self.acc_earth, self.gyro_earth, self.mag_earth = rotate_to_earth_frame(acc_body, gyro_body, mag_body)
        touchdowns = find_touchdowns(self.gyro_earth, self.freq)

        self.g_to_SI()

        velocity = find_velocity(self.acc_earth, touchdowns, self.freq)

        return self.result

    def g_to_SI(self):
        self.acc_earth *= 9.807     #[m/s/s]
        self.acc_earth -= [9.087, 0, 0]






