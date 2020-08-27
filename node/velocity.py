import numpy as np


def remove_drift(vel, touchdowns):
    """
    Remove drift after integrating
    :param vel: Integrated array [m/s]
    :param touchdowns: np.array with 0 where leg is in motion and 1 where leg is at the ground
    :return: vel: velocity array without drift [m/s]
    """
    # Set marker where motion start and stop
    diff = np.diff(np.asarray(touchdowns, dtype=int))
    start_motion = np.asarray(np.where(diff == -1)).flatten()  # Last index with zero value
    stop_motion = np.asarray(np.where(diff == 1)).flatten()  # Last index with non-zero value

    # Makes sure starting motion in the beginning, and stopping at the end
    if stop_motion[0] < start_motion[0]:
        start_motion = np.concatenate([[0], start_motion])

    if stop_motion[-1] < start_motion[-1]:
        stop_motion = np.concatenate([stop_motion, [len(vel) - 1]])

    # Remove drift
    for i in range(0, len(stop_motion)):
        # Find number of steps in the "motion"
        samples = stop_motion[i] - start_motion[i]

        # Find how much the drift is per sample vel[stop_motion[i]+1,:] == 0
        stepwise_drift = vel[stop_motion[i], :] / samples

        # Find drift to remove from each step in the motion
        drift = [stepwise_drift * i for i in range(1, samples + 1)]

        # Remove drift
        vel[start_motion[i] + 1:stop_motion[i] + 1, :] = vel[start_motion[i] + 1:stop_motion[i] + 1, :] - drift

    return vel


def find_velocity(acc, touchdowns, fs):
    """
    Integrate acceleration to get velocity. Removes drift after integration
    :param acc: acceleration [m/s/s]
    :param touchdowns: np.array with 0 where leg is in motion and 1 where leg is at the ground
    :param fs: sampling frequency [Hz]
    :return: vel: velocity [m/s]
    """
    vel = np.zeros((len(acc), 3))

    # Integrate acceleration
    for i in range(1, len(acc)):
        v0 = np.asarray(vel[i - 1, :])
        a = np.asarray(acc[i, :])
        vel[i, :] = v0 + a * 1 / fs

        # Force vel to be zero when foot is touching ground
        if touchdowns[i] != 0:
            vel[i, :] = [0, 0, 0]

    # Remove drift
    vel = remove_drift(vel, touchdowns)

    return vel
