# region Import packages and functions
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from node.result import Result
from node.rotation import rotate_to_earth_frame, change_coordinate_system
from node.touchdown import *
from node.velocity import *
from node.position import *
from node.orientation import find_orientation
from node.orientation1 import find_acc_lab_euler, find_acc_lab_quat
from node.animation import animate, plot_position
# endregion


class NodeAlgorithm:
    # Default feature columns (name of what the sensor measures)
    acc_body = np.empty([2, 2], dtype=float)
    acc_lab = np.empty([2, 2], dtype=float)
    gyro_body = np.empty([2, 2], dtype=float)
    mag_body = np.empty([2, 2], dtype=float)

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
        # region Preparation
        # Extract feature columns and change to new coordinate system, where z is down, y is in and x is forward
        # when sensor is placed with USB in and screws out on left foot
        self.acc_body = change_coordinate_system(
            np.asarray(df[['accX[mg]', 'accY[mg]', 'accZ[mg]']], dtype=float) * 10 ** -3)  # [g]
        self.gyro_body = change_coordinate_system(
            np.asarray(df[['gyroX[mdps]', 'gyroY[mdps]', 'gyroZ[mdps]']], dtype=float) * 10 ** -3)  # [dps]
        self.mag_body = change_coordinate_system(
            np.asarray(df[['magX[mG]', 'magY[mG]', 'magZ[mG]']], dtype=float) * 10 ** -3)  # [gauss]

        # Positive rotation counterclockwise
        self.gyro_body[:, 1] *= -1
        self.gyro_body[:, 2] *= -1

        # Apply low pass filter on data
        self.acc_body = filterLP(1, 5, self.freq, self.acc_body)
        self.gyro_body = filterLP(1, 5, self.freq, self.gyro_body)
        self.mag_body = filterLP(1, 5, self.freq, self.mag_body)

        # Remove gravity
        #self.remove_gravity(plot=False)
        # endregion

        # region Find touchdowns
        touchdowns_gyro = find_touchdowns_gyro(self.gyro_body, self.freq)
        touchdowns_acc = find_touchdowns_acc(self.acc_body, self.freq)

        touchdowns = touchdowns_gyro
        # Compare touchdowns
        # compare_touchdowns(touchdowns_gyro, touchdowns_acc, self.acc_body, self.gyro_body, self.freq,
        #                   name1='gyro-touchdowns', name2='acc-touchdowns')

        # endregion


        # region Find orientation
        acc_lab_euler = find_acc_lab_euler(self.acc_body, self.gyro_body, self.freq)
        acc_lab_quat = find_acc_lab_quat(self.acc_body, self.gyro_body, self.mag_body, self.freq)
        self.acc_lab = acc_lab_quat
        # endregion

        # region Find velocity
        # Transform acc data to have SI units
        self.g_to_SI()

        velocity_manual = find_velocity(self.acc_lab, touchdowns, self.freq, plot_drift=True, plot_vel=True)
        velocity_cumtrapz = find_velocity_cumtrapz(self.acc_lab, self.freq, plot_vel=True)
        # endregion

        # region Find position
        # First method: integrate velocity manually
        position_manual = find_position(velocity_manual, touchdowns, self.freq, plot_drift=False)
        # animate(position_manual)  # Animation does not look good. Christ

        # Second method: integrate using cumtrapz. By using velocity_cumtrapz, position_cumtrapz increase rapidly,
        # but by using velocity_manual result is basically the same as pos_man
        position_cumtrapz = find_position_cumtrapz(velocity_cumtrapz, touchdowns, self.freq, plot_drift=False)

        #animate(position_cumtrapz, save=True)  # Animation does not look good. Christ

        fo = 100     # Frequency of showing orientation in position plot [Hz]
        plot_position(position_manual, self.freq)
        # endregion

        return self.result

    def g_to_SI(self):
        """
        Convert acceleration data to SI units and linear acceleration (remove gravity)
        :return:
        """
        self.acc_body *= 9.807  # [m/s/s]
        self.acc_lab *= 9.807  # [m/s/s]

    def remove_gravity(self, plot=False):
        bias = np.mean(self.acc_body[:self.freq * 1, :], axis=0)
        # Remove bias (mostly earth gravity)
        old_acc = self.acc_body
        self.acc_body = self.acc_body - bias

        if plot:
            time = np.linspace(0, len(old_acc) / self.freq, len(old_acc))
            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True)
            ax1.set_title("x direction")
            ax1.plot(time, old_acc[:, 0], 'b', label="Acceleration")
            ax1.plot(time, self.acc_body[:, 0], 'g', label="Linear acceleration")
            ax1.legend(loc=1)

            ax2.set_title("y direction")
            ax2.plot(time, old_acc[:, 1], 'b', label="Acceleration")
            ax2.plot(time, self.acc_body[:, 1], 'g', label="Linear acceleration")
            ax2.legend(loc=1)

            ax3.set_title("z direction")
            ax3.plot(time, old_acc[:, 2], 'b', label="Acceleration")
            ax3.plot(time, self.acc_body[:, 2], 'g', label="Linear acceleration")
            ax3.legend(loc=1)

            fig.text(0.5, 0.04, 'Time [s]', ha='center')
            fig.text(0.04, 0.5, 'Acceleration [g]', va='center', rotation='vertical')
            plt.show()




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

