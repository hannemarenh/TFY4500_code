import numpy as np
import pandas as pd
from node.rotation import rotation_matrix_from_vectors
from scipy import integrate
import matplotlib.pyplot as plt
from scipy.fft import fft


def fourier(y, fs):
    N = len(y)
    # sample spacing
    T = 1.0 / fs
    x = np.linspace(0.0, N * T, N)
    yf = fft(y)
    xf = np.linspace(0.0, 1.0 / (2.0 * T), N // 2)
    plt.plot(xf, 2.0 / N * np.abs(yf[0:N // 2]))
    plt.grid()
    plt.show()
    return xf, yf


# Load df made in current_df.py
df = pd.read_pickle('current_df.pkl')

acc = np.asarray(df[['accX[mg]', 'accY[mg]', 'accZ[mg]']], dtype=float)*10**-3
gyro = np.asarray(df[['gyroX[mdps]', 'gyroY[mdps]', 'gyroZ[mdps]']], dtype=float)*10**-3
mag = np.asarray(df[['magX[mG]', 'magY[mG]', 'magZ[mG]']], dtype=float)*10**-3


# Rotate to earth frame
# Want to change rotation to class such that the rotation itself is done in rotation.py, and not here
acc_earth = [1, 0, 0]
acc_body = acc[0, :]

rot = rotation_matrix_from_vectors(acc_body, acc_earth)
for i in range(len(acc[:, 0])):
    acc[i, :] = rot.dot(acc[i, :])
    gyro[i, :] = rot.dot(gyro[i, :])
    mag[i, :] = rot.dot(mag[i, :])

dt = 1/100
x = integrate.cumtrapz(integrate.cumtrapz(acc[:, 0], dx=dt, initial=0), dx=dt, initial=0)
y = integrate.cumtrapz(integrate.cumtrapz(acc[:, 1], dx=dt, initial=0), dx=dt, initial=0)
z = integrate.cumtrapz(integrate.cumtrapz(acc[:, 2], dx=dt, initial=0), dx=dt, initial=0)

# Plot 3D Trajectory
fig3, ax = plt.subplots()
fig3.suptitle('3D Trajectory of leg')
ax = plt.axes(projection='3d')
ax.plot3D(x, y, z, c='red', lw=5, label='leg trajectory')
ax.set_xlabel('X position (m)')
ax.set_ylabel('Y position (m)')
ax.set_zlabel('Z position (m)')
plt.show()