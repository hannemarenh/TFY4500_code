import numpy as np
from scipy import signal
import math
from node.gait_tracking import ahrs
from node.gait_tracking.quaternion_utils import quaternion_rotate, quaternion_conjugate, quaternion_to_rotmat
import pandas as pd
from node.animation import *


class GaitTracker:
    def __init__(self, hz, highpass, lowpass, initial_orientation=None):
        if initial_orientation is None:
            initial_orientation = [1, 0, 0, 0]

        self.hz = hz
        self.highpass = highpass
        self.lowpass = lowpass
        self.initial_orientation = initial_orientation

    def calculate(self, df, detect_stationary=True):
        """
        :param df DataFrame used to calculate position and orientation.
        Assumes columns ["acc_x", "acc_y", "acc_z", "gyro_x, "gyro_y", "gyro_z"]

        :return: dictionary containing orientation, acceleration, velocity and position
        """
        self.detect_stationary_periods(df)

        quat = self.compute_orientation(df)

        acc = self.calculate_acceleration(df, quat)

        vel = self.calculate_velocity(df, acc)

        pos = self.calculate_position(vel)

        return {
            "orientation": quat,
            "acceleration": acc,
            "velocity": vel,
            "position": pos
        }      

    def detect_stationary_periods(self, df, cutoff=0.07):
        # magnitude of acceleration
        df["acc_mag"] = np.sqrt(df["acc_x"] ** 2 + df["acc_y"] ** 2 + df["acc_z"] ** 2)

        # highpass filter accel
        filter_cutoff = self.highpass
        b, a = signal.butter(1, (2 * filter_cutoff) / self.hz, 'high')

        acc_mag_filt = signal.filtfilt(b, a, df["acc_mag"])

        # absolute value of filter
        acc_mag_filt = np.abs(acc_mag_filt)

        # lowpass filter accel
        filter_cutoff = self.lowpass
        b, a = signal.butter(1, (2 * filter_cutoff) / self.hz, 'low')
        acc_mag_filt = signal.filtfilt(b, a, acc_mag_filt)

        df["stationary"] = acc_mag_filt < cutoff

    def compute_orientation(self, df) -> np.array:
        """
        Computes orientation using AHRS algorithm. 
        
        :param df: Accelerometer and Gyroscope data
        :type df: pandas.DataFrame

        :returns: Orientation for each timestep expressed as quaternion
        :rtype: numpy.array
        """
        init_period = 2
        #ran = range(df.index.min(), df[df["time"] <= df.iloc[0]["time"] + init_period].index.max())
        acc = df[["acc_x","acc_y", "acc_z"]].to_numpy()
        gyro = df[["gyro_x", "gyro_y", "gyro_z"]].to_numpy()
        stationary = None
        if "stationary" in df.columns:
            stationary = df["stationary"].to_numpy()
        
        return self.__compute_orientation(acc, gyro, stationary)

    def __compute_orientation(self, acc, gyro, ran=None, stationary=None):
        # Computed using AHRS (attitude and heading reference system) algorithm
        quat = []
        alg = ahrs.AHRS(sample_period=1 / self.hz, kp=1, kp_init=1, initial_orientation=self.initial_orientation)

        # initial convergence
        #for i in range(0, 2000):
        #    alg.update_imu(np.array([0, 0, 0]),
        #                   np.array([np.mean(acc[ran, 0]), np.mean(acc[ran, 1]), np.mean(acc[ran, 2])]))

        # for all data
        for i in np.arange(len(acc)):
            if stationary is None:
                alg.kp = 0
            elif stationary[i]:
                alg.kp = 0.5

            alg.update_imu(gyro[i, :] * math.pi / 180, acc[i, :])

            quat.append(alg.quaternion())

        return np.array(quat)

    def calculate_acceleration(self, df, quat):
        # Rotate body accelerations to Earth frame
        a = df[["acc_x", "acc_y", "acc_z"]].values.astype(float)
        R = quaternion_to_rotmat(quat)
        acc = np.zeros((len(df['acc_x']),3))
        for i in range(len(df['acc_x'])):
            acc[i, :] = R[:, :, i].dot(a[i, :])
        #acc = quaternion_rotate(df[["acc_x", "acc_y", "acc_z"]].values.astype(float), quaternion_conjugate(quat))
        # Remove gravity from measurements
        gravity = np.array([0, 0, 1])
        acc -= gravity
        return acc*9.81

    def calculate_velocity(self, df, acc):
        sample_period = 1 / self.hz

        # Integrate acceleration to yield velocity
        vel = np.zeros(acc.shape)
        for t in range(1, acc.shape[0]):
            vel[t, :] = vel[t - 1, :] + acc[t, :] * sample_period
            if df.iloc[t]["stationary"]:
                vel[t, :] = np.zeros(3)  # force zero velocity when foot stationary

        # Compute integral drift during non-stationary periods
        vel_drift = np.zeros(vel.shape)

        d = np.append(arr=[0], values=np.diff(df["stationary"].astype(np.int8)))
        stationary_start = np.where(d == -1)
        stationary_end = np.where(d == 1)
        stationary_start = np.array(stationary_start)[0]
        stationary_end = np.array(stationary_end)[0]

        for i in range(len(stationary_end)):
            # velocity just before stationary period divided by length of period
            drift_rate = vel[stationary_end[i] - 1, :] / (stationary_end[i] - stationary_start[i])
            enum = np.arange(1, stationary_end[i] - stationary_start[i] + 1)
            enum_t = enum.reshape((1, len(enum)))
            drift_rate_t = drift_rate.reshape((1, len(drift_rate)))
            # multiply drift rate with corresponding data index to get drift value at each point
            drift = enum_t.T * drift_rate_t

            vel_drift[stationary_start[i]:stationary_end[i], :] = drift

        # compute linear velocity by subtracting the drift at each point
        return vel - vel_drift

    def calculate_position(self, vel):
        sample_period = 1 / self.hz

        # Integrate velocity to yield position
        pos = np.zeros(vel.shape)
        for t in range(1, pos.shape[0]):
            pos[t, :] = pos[t - 1, :] + vel[t, :] * sample_period

        return pos

if __name__ == '__main__':
    df = pd.read_csv('spiralStairs_CalInertialAndMag.csv')
    df.columns = (["Packet number", "gyro_x", "gyro_y", "gyro_z", "acc_x", "acc_y", "acc_z", "mag_x", "mag_y", "mag_z"])

    test = GaitTracker(hz=256, highpass=0.001, lowpass=5)
    test.calculate(df)