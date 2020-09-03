#region Import packages and functions
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from node.result import Result
from node.rotation import rotate_to_earth_frame
from node.touchdown import *
from node.velocity import *
from node.position import *
from node.animation import animate

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
        self.acc_earth, self.gyro_earth, self.mag_earth = rotate_to_earth_frame(acc_body, gyro_body, mag_body,
                                                                                self.freq, plot_rotation=True)
        # Transform acc data to have SI units
        self.g_to_SI()

        # Find touchdowns
        touchdowns = find_touchdowns_gyro(self.gyro_earth, self.freq)
        touchdowns_acc = find_touchdowns_acc(self.acc_earth, self.freq)

        # Compare touchdowns
        #compare_touchdowns(touchdowns, touchdowns_acc, self.acc_earth, self.gyro_earth, self.freq,
        #                   name1='gyro-touchdowns', name2='acc-touchdowns')

        # Find velocity
        velocity = find_velocity(self.acc_earth, touchdowns, self.freq, plot_drift=False, plot_vel=True)
        velocity_cumtrapz = find_velocity_cumtrapz(self.acc_earth, self.freq, plot_vel=False)

        # Find position
        position = find_position(velocity, touchdowns, self.freq, plot_drift=False)
        animate(position)
        position_cumtrapz = find_position_cumtrapz(velocity_cumtrapz, touchdowns, self.freq, plot_drift=True)
        animate(position_cumtrapz)
        np.savetxt(r"C:\\Users\\Hanne Maren\\Documents\\nivo\\python-experiments\\pos.csv", position, delimiter=",")


        return self.result

    def g_to_SI(self):
        """
        Convert acceleration data to SI units and linear acceleration (remove gravity)
        :return:
        """
        self.acc_earth *= 9.807     #[m/s/s]
        self.acc_earth -= [9.087, 0, 0]

    def plot(self):
        """
        Plot measured data
        """
        # Make time axis
        size = len(self.acc_earth)
        time = np.linspace(0, size / self.freq, size)

        fig, (ax1, ax2, ax3) = plt.subplots(3)
        ax1.plot(time, self.acc_earth[:, 0], label='accX')
        ax1.plot(time, self.acc_earth[:, 1], label='accY')
        ax1.plot(time, self.acc_earth[:, 2], label='accZ')
        ax1.set_xlabel('Time [s]')
        ax1.set_ylabel('Acceleration [g]')
        ax1.legend(loc='upper right')

        ax2.plot(time, self.gyro_earth[:, 0], label='gyroX')
        ax2.plot(time, self.gyro_earth[:, 1], label='gyroY')
        ax2.plot(time, self.gyro_earth[:, 2], label='gyroZ')
        ax2.set_xlabel('Time [s]')
        ax2.set_ylabel('Gyroscope data [dps]')
        ax2.legend(loc='upper right')

        ax3.plot(time, self.mag_earth[:, 0], label='magX')
        ax3.plot(time, self.mag_earth[:, 1], label='magY')
        ax3.plot(time, self.mag_earth[:, 2], label='magZ')
        ax3.set_xlabel('Time [s]')
        ax3.set_ylabel('Magnetometer data [G]')
        ax3.legend(loc='upper right')

        plt.show()






