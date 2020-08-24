import pandas as pd
import numpy as np
from scipy import signal
from matplotlib import pyplot as plt


def trapezoid(y0, x0, x1, dt):
    """
    Trapezoid integration

    Inputs:
    y0: previous integrated point
    x0: previous point in array to be integrated
    x1: current point in array to be integrated
    dt: sampling time

    Output:
    Next integrated point
    """
    return y0 + (x0 + x1)*dt/2


def filterHP(order, cutOff, fs, array):
    """
    High pass filtering of array

    Inputs:
    order       - filter order
    cutoff      - cutoff frequency in units of Nyquist
    fs          - sampling frequency

    Output:
    arrayHP     - high pass filtered array
    """
    # Define sizes of numpy array
    rows = acc.shape[0]
    cols = acc.shape[1]

    # Initialize high pass filtered array
    arrayHP = np.zeros((rows, cols))

    # Highpass filtering to remove baseline drift
    [b, a] = signal.butter(order, (2 * cutOff) / fs, 'highpass')

    for j in range(0, cols):
        arrayHP[:, j] = signal.filtfilt(b, a, array[:, j])

    return arrayHP


def filterLP(order, cutOff, fs, array):
    """
    Low pass filtering of array

    Inputs:
    order       - filter order
    cutoff      - cutoff frequency in units of Nyquist
    fs          - sampling frequency

    Output:
    arrayLP     - low pass filtered array
    """
    # Define sizes of numpy array
    rows = acc.shape[0]
    cols = acc.shape[1]

    # Initialize high pass filtered array
    arrayLP = np.zeros((rows, cols))

    # Low pass filtering to remove small oscillations
    [b, a] = signal.butter(order, (2 * cutOff) / fs, 'lowpass')

    for j in range(0, cols):
        arrayLP[:, j] = signal.filtfilt(b, a, array[:, j])

    return arrayLP


#Load csv file. Skip lines where measurements are missing
title = r"output.csv"
file = r"C:\Users\Hanne Maren\Documents\nivo\project\legData\horseLegs\frontRight\\" + title


df = pd.read_csv(file, error_bad_lines=False)

# Remove "forskyvede" rows
# Dont know why they occur, but they do... :( Probably something with the sensor
# For sensorTile
check = df.iloc[:, -1].notnull()
for i in range(0, len(check)):
    if check[i]:
        df = df.drop(i)

# Delete nan rows
df = df.iloc[:, :-1].dropna(axis=0)

acc = np.asarray(df[['accX[mg]', 'accY[mg]', 'accZ[mg]']],dtype=float)*10**-3
gyro = np.asarray(df[['gyroX[mdps]', 'gyroY[mdps]', 'gyroZ[mdps]']],dtype=float)*10**-3
mag = np.asarray(df[['magX[mG]', 'magY[mG]', 'magZ[mG]']],dtype=float)*10**-3

#region Plot raw data
start = 18000
stop = 20000
acc_raw = acc[start:stop]
gyro_raw = gyro[start:stop]
mag_raw = mag[start:stop]

# Make time axis
freq = 100  #[Hz]
size = len(acc)
time = np.linspace(0, size / (freq), size)


fig, (ax1, ax2, ax3) = plt.subplots(3)
ax1.plot(time, acc_raw[:, 0], label='accX')
ax1.plot(time, acc_raw[:, 1], label='accY')
ax1.plot(time, acc_raw[:, 2], label='accZ')
ax1.set_xlabel('Time [s]')
ax1.set_ylabel('Acceleration [g]')
ax1.legend(loc='upper right')

ax2.plot(time, gyro_raw[:, 0], label='gyroX')
ax2.plot(time, gyro_raw[:, 1], label='gyroY')
ax2.plot(time, gyro_raw[:, 2], label='gyroZ')
ax2.set_xlabel('Time [s]')
ax2.set_ylabel('Gyroscope data [dps]')
ax2.legend(loc='upper right')

ax3.plot(time, mag_raw[:, 0], label='magX')
ax3.plot(time, mag_raw[:, 1], label='magY')
ax3.plot(time, mag_raw[:, 2], label='magZ')
ax3.set_xlabel('Time [s]')
ax3.set_ylabel('Magnetometer data [G]')
ax3.legend(loc='upper right')

plt.show()
#endregion
