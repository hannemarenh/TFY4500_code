import numpy as np
from numpy import linalg as LA
from node.gait_tracking.quaternion_utils import quaternion_conjugate, quaternion_product

class AHRS:
    def __init__(self, sample_period, kp=2, kp_init=200, beta=0.1, initial_orientation=None):
        if initial_orientation is None:
            initial_orientation = [1, 0, 0, 0]

        self.sample_period = sample_period
        self.kp = kp  # proportional gain
        self.ki = 0  # integral gain
        self.kp_init = kp_init  # proportional gain used during initialisation
        self.init_period = 5  # initialization period in seconds

        self._q = np.array(initial_orientation)  # internal quaternion describing the Earth relative to the sensor
        self._int_error = np.array([0, 0, 0])  # integral error
        self._kp_ramped = 0  # internal proportional gain used to ramp during initialisation
        
        self.beta = beta # algorithm gain
        
    def quaternion(self):

        return self._q

    def update(self, gyroscope, accelerometer, magnetometer):
        q = self._q
        
        # normalize accelerometer measurement
        if np.isnan(LA.norm(accelerometer)) or LA.norm(accelerometer) == 0:
            # handle NA values
            print("Accelerometer magnitude is zero. Algorithm update aborted")
            return
        else:
            # normalize measurement
            accelerometer /= LA.norm(accelerometer)
            
        # normalize magnetometer measurement
        if np.isnan(LA.norm(magnetometer)) or LA.norm(magnetometer) == 0:
            # handle NA values
            print("Magnetometer magnitude is zero. Algorithm update aborted")
            return
        else:
            # normalize measurement
            magnetometer /= LA.norm(magnetometer)
            
        # reference direction of Earths magnetic field
        h = quaternion_product(q, quaternion_product(np.insert(magnetometer, 0, 0), quaternion_conjugate(q)))
        b = np.array([0, LA.norm([h[1], h[2]]), 0, h[3]])
        
        # gradient decent algorithm corrective step
        F = np.array([
            2 * (q[1] * q[3] - q[0] * q[2]) - accelerometer[0],
            2 * (q[0] * q[1] + q[2] * q[3]) - accelerometer[1],
            2 * (0.5 - q[1]**2 - q[2]**2) - accelerometer[2],
            2 * b[1] * (0.5 - q[2]**2 - q[3]**2) + 2 * b[3] * (q[1] * q[3] - q[0] * q[2]) - magnetometer[0],
            2 * b[1] * (q[1] * q[2] - q[0] * q[3]) + 2 * b[3] * (q[0] * q[1] + q[2] * q[3]) - magnetometer[1],
            2 * b[1] * (q[0] * q[2] + q[1] * q[3]) + 2 * b[3] * (0.5 - q[1]**2 - q[2]**2) - magnetometer[2]
        ])
        
        J = np.array([
            [-2 * q[2], 2 * q[3], -2 * q[0], 2 * q[1]],
            [2 * q[1], 2 * q[0], 2 * q[3], 2 * q[2]],
            [0, -4 * q[1], -4 * q[2], 0],
            [-2 * b[3] * q[2], 2 * b[3] * q[3], -4 * b[1] * q[2] - 2 * b[3] * q[0], -4 * b[1] * q[3] + 2 * b[3] * q[1]],
            [-2 * b[1] * q[3] + 2 * b[3] * q[1], 2 * b[1] * q[2] + 2 * b[3] * q[1], 2 * b[1] * q[1] + 2 * b[3] * q[3], -2 * b[1] * q[0] + 2 * b[3] * q[2]],
            [2 * b[1] * q[2], 2 * b[1] * q[3] - 4 * b[3] * q[1], 2 * b[1] * q[0] - 4 * b[3] * q[2], 2 * b[1] * q[1]]
        ])
        
        step = J.T @ F.reshape(-1, 1)
        step /= LA.norm(step)
        
        q_dot = 0.5 * (quaternion_product(q, np.insert(gyroscope, 0, 0))) - self.beta * step.T
        
        q = q + q_dot[0] * self.sample_period
        self._q = q / LA.norm(q)        
    
    def update_imu(self, gyroscope, accelerometer):
        q = self._q
        
        # normalize accelerometer measurement
        if np.isnan(LA.norm(accelerometer)) or LA.norm(accelerometer) == 0:
            # handle NA values
            print("Accelerometer magnitude is zero. Algorithm update aborted")
            return
        else:
            # normalize measurement
            accelerometer /= LA.norm(accelerometer)
        
        F = np.array([
            2 * (q[1] * q[3] - q[0] * q[2]) - accelerometer[0],
            2 * (q[0] * q[1] + q[2] * q[3]) - accelerometer[1],
            2 * (0.5 - q[1]**2 - q[2]**2) - accelerometer[2]
        ])
        J = np.array([
            [-2 * q[2], 2 * q[3], -2 * q[0], 2 * q[1]],
            [2 * q[1], 2 * q[0], 2 * q[3], 2 * q[2]],
            [0, -4 * q[1], -4 * q[2], 0]
        ])
        step = J.T @ F.reshape(-1, 1)
        step /= LA.norm(step)
        
        q_dot = 0.5 * (quaternion_product(q, np.insert(gyroscope, 0, 0))) - self.beta * step.T
        
        q = q + q_dot[0] * self.sample_period
        self._q = q / LA.norm(q)

    def reset(self):
        self._kp_ramped = self.kp_init  # start Kp ramp-down
        self._int_error = np.array([0, 0, 0])  # reset integral terms
        self._q = np.array([1, 0, 0, 0])  # set quaternion to alignment
