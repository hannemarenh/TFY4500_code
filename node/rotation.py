import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def rotation_matrix_from_vectors(vec1, vec2):
    """ Find the rotation matrix that aligns vec1 to vec2
    :param vec1: A 3d "source" vector
    :param vec2: A 3d "destination" vector
    :return mat: A transform matrix (3x3) which when applied to vec1, aligns it with vec2.
    """
    a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
    return rotation_matrix


# Load df made in current_df.py
df = pd.read_pickle('current_df.pkl')

acc = np.asarray(df[['accX[mg]', 'accY[mg]', 'accZ[mg]']], dtype=float)*10**-3
gyro = np.asarray(df[['gyroX[mdps]', 'gyroY[mdps]', 'gyroZ[mdps]']], dtype=float)*10**-3
mag = np.asarray(df[['magX[mG]', 'magY[mG]', 'magZ[mG]']], dtype=float)*10**-3

start = 18000
stop = 20000
acc = acc[start:stop]
gyro = gyro[start:stop]
mag = mag[start:stop]

acc_earth = [1, 0, 0]
acc_body = acc[0, :]

rot = rotation_matrix_from_vectors(acc_body, acc_earth)

acc_check = acc.copy()
gyro_check = gyro.copy()
mag_check = mag.copy()

for i in range(len(acc[:, 0])):
    acc[i, :] = rot.dot(acc[i, :])
    gyro[i, :] = rot.dot(gyro[i, :])
    mag[i, :] = rot.dot(mag[i, :])


#region Plot
fig, ((ax1, ax2, ax3), (ax4, ax5, ax6),(ax7,ax8,ax9)) = plt.subplots(3,3)
ax1.set_title("AccX")
ax1.set_ylabel("g")
ax1.plot(acc_check[:, 0], 'r', label="Body frame")
ax1.plot(acc[:, 0], 'g', label="Earth frame")
ax1.legend(loc=1)

ax2.set_title("AccY")
ax2.set_ylabel("g")
ax2.plot(acc_check[:, 1], 'r', label="Body frame")
ax2.plot(acc[:, 1], 'g', label="Earth frame")
ax2.legend(loc=1)

ax3.set_title("AccZ")
ax3.set_ylabel("g")
ax3.plot(acc_check[:, 2], 'r', label="Body frame")
ax3.plot(acc[:, 2], 'g', label="Earth frame")
ax3.legend(loc=1)

ax4.set_title("GyroX")
ax4.set_ylabel("dps")
ax4.plot(gyro_check[:, 0], 'r', label="Body frame")
ax4.plot(gyro[:, 0], 'g', label="Earth frame")
ax4.legend(loc=1)

ax5.set_title("GyroY")
ax5.set_ylabel("dps")
ax5.plot(gyro_check[:, 1], 'r', label="Body frame")
ax5.plot(gyro[:, 1], 'g', label="Earth frame")
ax5.legend(loc=1)

ax6.set_title("GyroZ")
ax6.set_ylabel("dps")
ax6.plot(gyro_check[:, 2], 'r', label="Body frame")
ax6.plot(gyro[:, 2], 'g', label="Earth frame")
ax6.legend(loc=1)

ax7.set_title("MagX")
ax7.set_ylabel("gauss")
ax7.plot(mag_check[:, 0], 'r', label="Body frame")
ax7.plot(mag[:, 0], 'g', label="Earth frame")
ax7.legend(loc=1)

ax8.set_title("MagY")
ax8.set_ylabel("gauss")
ax8.plot(mag_check[:, 1], 'r', label="Body frame")
ax8.plot(mag[:, 1], 'g', label="Earth frame")
ax8.legend(loc=1)

ax9.set_title("MagZ")
ax9.set_ylabel("gauss")
ax9.plot(mag_check[:, 2], 'r', label="Body frame")
ax9.plot(mag[:, 2], 'g', label="Earth frame")
ax9.legend(loc=1)
#endregion