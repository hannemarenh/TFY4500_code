import numpy as np
from scipy.spatial.transform import Rotation as R


def quaternion_product(q1, q2):
    if len(q1.shape) > 1:
        w1 = q1[:, 0]
        x1 = q1[:, 1]
        y1 = q1[:, 2]
        z1 = q1[:, 3]

        w2 = q2[:, 0]
        x2 = q2[:, 1]
        y2 = q2[:, 2]
        z2 = q2[:, 3]

        res = np.zeros(((w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2).shape[0], 4))
        res[:, 0] = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        res[:, 1] = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        res[:, 2] = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
        res[:, 3] = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

        return res
    else:
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2

        return np.array([
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
        ])


def quaternion_conjugate(q):
    if len(q.shape) > 1:
        res = np.zeros(q.shape)
        res[:, 0] = q[:, 0]
        res[:, 1] = -q[:, 1]
        res[:, 2] = -q[:, 2]
        res[:, 3] = -q[:, 3]

        return res
    else:
        return np.array([q[0], -q[1], -q[2], -q[3]])


def quaternion_rotate(v, q):
    q = q.reshape(-1, 4)
    acc_quaternion = np.insert(v, 0, 0, axis=1)

    v0XYZ = quaternion_product(quaternion_product(q, acc_quaternion), quaternion_conjugate(q))
    return v0XYZ[:, 1:4]


def quaternion_to_rotmat(q):
    result = np.zeros((3, 3, q.shape[0]))
    result[0, 0, :] = 2 * q[:, 0] ** 2 - 1 + 2 * q[:, 1] ** 2
    result[0, 1, :] = 2 * (q[:, 1] * q[:, 2] + q[:, 0] * q[:, 3])
    result[0, 2, :] = 2 * (q[:, 1] * q[:, 3] - q[:, 0] * q[:, 2])
    result[1, 0, :] = 2 * (q[:, 1] * q[:, 2] - q[:, 0] * q[:, 3])
    result[1, 1, :] = 2 * q[:, 0] ** 2 - 1 + 2 * q[:, 2] ** 2
    result[1, 2, :] = 2 * (q[:, 2] * q[:, 3] + q[:, 0] * q[:, 1])
    result[2, 0, :] = 2 * (q[:, 1] * q[:, 3] + q[:, 0] * q[:, 2])
    result[2, 1, :] = 2 * (q[:, 2] * q[:, 3] - q[:, 0] * q[:, 1])
    result[2, 2, :] = 2 * q[:, 0] ** 2 - 1 + 2 * q[:, 3] ** 2
    return result


def rotation_matrix(yaw, pitch, roll, degrees=False):
    if degrees:
        yaw, pitch, roll = np.deg2rad(yaw), np.deg2rad(pitch), np.deg2rad(roll)
    r_x = np.array([[1, 0, 0], [0, np.cos(yaw), -np.sin(yaw)], [0, np.sin(yaw), np.cos(yaw)]])
    r_y = np.array([[np.cos(pitch), 0, np.sin(pitch)], [0, 1, 0], [-np.sin(pitch), 0, np.cos(pitch)]])
    r_z = np.array([[np.cos(roll), -np.sin(roll), 0], [np.sin(roll), np.cos(roll), 0], [0, 0, 1]])

    return r_x @ r_y @ r_z


def scalar_last_to_scalar_first(q):
    return np.array([q[3], q[0], q[1], q[2]])


def orientation_as_quat(x, y, z, degrees=False):
    return scalar_last_to_scalar_first(R.from_matrix(rotation_matrix(x, y, z, degrees)).as_quat())
