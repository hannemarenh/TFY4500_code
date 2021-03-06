import pandas as pd
import numpy as np
from scipy import signal, fftpack
from matplotlib import pyplot as plt
from scipy.fft import fft


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


def filterHP(order, cutOff, fs, array, plotFilter=False):
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
    rows = array.shape[0]
    if array.ndim == 1:
        # Initialize high pass filtered array
        arrayHP = np.zeros(rows)

        # Low pass filtering to remove small oscillations
        [b, a] = signal.butter(order, (2 * cutOff) / fs, 'highpass')

        arrayHP = signal.filtfilt(b, a, array)

    else:
        cols = array.shape[1]

        # Initialize high pass filtered array
        arrayHP = np.zeros((rows, cols))

        # Low pass filtering to remove small oscillations
        [b, a] = signal.butter(order, (2 * cutOff) / fs, 'highpass')

        for j in range(0, cols):
            arrayHP[:, j] = signal.filtfilt(b, a, array[:, j])

    if plotFilter:
        # Plot the frequency response
        w, h = signal.freqz(b, a)
        plt.plot((fs * 0.5 / np.pi) * w, 20 * np.log10(abs(h)), label="order = %d" % order)
        plt.axvline(cutOff, color='red')  # cutoff frequency
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Gain [dB]')
        plt.grid(True)
        plt.legend(loc='best')
        plt.title("Butterworth filter frequency response")
        plt.show()

    return arrayHP


def filterLP(order, cutOff, fs, array, plotFilter=False):
    """
    Low pass filtering of array

    Inputs:
    order       - filter order
    cutoff      - cutoff frequency [Hz]
    fs          - sampling frequency  [Hz]
    plotFilter  - true/false

    Output:
    arrayLP     - low pass filtered array
    """
    # Define sizes of numpy array
    rows = array.shape[0]
    if array.ndim == 1:
        # Initialize high pass filtered array
        arrayLP = np.zeros(rows)

        # Low pass filtering to remove high frequencies
        b, a = signal.butter(order, (2 * cutOff) / fs, 'lowpass', output='ba')

        arrayLP = signal.filtfilt(b, a, array)

    else:
        cols = array.shape[1]

        # Initialize low pass filtered array
        arrayLP = np.zeros((rows, cols))

        # Low pass filtering to remove high frequencies
        b, a = signal.butter(order, (2 * cutOff) / fs, 'lowpass', output='ba')

        for j in range(0, cols):
            arrayLP[:, j] = signal.filtfilt(b, a, array[:, j])

    if plotFilter:
        # Plot the frequency response
        w, h = signal.freqz(b, a)
        plt.plot((fs * 0.5 / np.pi) * w, 20*np.log10(abs(h)), label="order = %d" % order)
        plt.axvline(cutOff, color='green', label="Cutoff frequency")  # cutoff frequency
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Gain [dB]')
        plt.grid(True)
        plt.legend(loc='best')
        plt.title("Butterworth filter frequency response")
        plt.show()

    return arrayLP


def plot_raw_data(start, stop, acc, gyro, mag):
    """
    Plots measured raw data.
    :param start: first sample number to plot
    :param stop: last sample number to plot
    :param acc: raw acceleration data [g]
    :param gyro: raw angular rate data [dps]
    :param mag: raw magnetization data [G]
    :return acc, gyro, mag, time: Shortened arrays for acc, gyro and mag if
    """
    acc = acc[start:stop]
    gyro = gyro[start:stop]
    mag = mag[start:stop]

    # Make time axis
    freq = 100  #[Hz]
    size = len(acc)
    time = np.linspace(0, size / freq, size)

    fig, (ax1, ax2, ax3) = plt.subplots(3)
    ax1.plot(time, acc[:, 0], label='accX')
    ax1.plot(time, acc[:, 1], label='accY')
    ax1.plot(time, acc[:, 2], label='accZ')
    ax1.set_xlabel('Time [s]')
    ax1.set_ylabel('Acceleration [g]')
    ax1.legend(loc='upper right')

    ax2.plot(time, gyro[:, 0], label='gyroX')
    ax2.plot(time, gyro[:, 1], label='gyroY')
    ax2.plot(time, gyro[:, 2], label='gyroZ')
    ax2.set_xlabel('Time [s]')
    ax2.set_ylabel('Gyroscope data [dps]')
    ax2.legend(loc='upper right')

    ax3.plot(time, mag[:, 0], label='magX')
    ax3.plot(time, mag[:, 1], label='magY')
    ax3.plot(time, mag[:, 2], label='magZ')
    ax3.set_xlabel('Time [s]')
    ax3.set_ylabel('Magnetometer data [G]')
    ax3.legend(loc='upper right')

    plt.show()
    return acc, gyro, mag, time


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


def find_touchdowns_gyro(gyro, fs):
    """
    Find touchdowns(=when horse leg is on the ground) based on gyroscope data. in y direction.
    :param gyro: gyroscope data
    :return: touchdowns: np.array with 0 where leg is in motion and 1 where leg is at the ground
    """
    # Absolute value of gradient of gyro
    dgyro_abs = np.abs(np.gradient(gyro, axis=0))

    # Filter dgyro
    order = 1
    cutOff = 10
    lp_dgyro_abs = filterLP(order, cutOff, fs, dgyro_abs)

    threshold = 3
    touchdowns = np.asarray(lp_dgyro_abs[:, 1] <= threshold, dtype=int)

    return touchdowns


def find_touchdowns_acc(acc, fs):
    """
    :param acc: acceleration with shape (len, 3) [m/s/s]
    :param fs: sampling frequency
    :return: touchdowns: np.array with 0 where leg is in motion and 1 where leg is at the ground
    """
    threshold = 0.25
    env = np.abs(signal.hilbert(acc[:, 2]))
    touchdowns = np.asarray(env <= threshold, dtype=int)

    return touchdowns


def find_touchdown_ai(acc, gyro, mag, fs):
    touchdowns = np.zeros(acc.shape)

    return touchdowns


def compare_touchdowns(method1, method2, acc, gyro, fs, name1='method 1', name2='method2'):
    """
    Compare two methods for finding touchdowns by plotting them toghether, with acc and with gyro
    :param method1: np.array (true/false) with touchdowns found by method 1
    :param method2: np.array (true/false) with touchdowns found by method 2
    :param acc: acceleration data
    :param gyro: gyroscope data
    :param fs: sampling frequency
    :param name1: Name of first method
    :param name2: Name of second method
    :return: 0
    """
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1)

    ax1.plot(method1, 'y', label=name1)
    ax1.plot(method2, 'm', label=name2)
    ax1.legend(loc=0)

    ax2.plot(acc[:, 0], 'g--', label="accX")
    ax2.plot(acc[:, 1], 'b--', label="accY")
    ax2.plot(acc[:, 2], 'c--', label="accZ")
    ax2.plot(method1*acc.max(), 'y', label=name1)
    ax2.plot(method2*acc.max(), 'm', label=name2)
    ax2.legend(loc=0)

    ax3.plot(gyro[:, 0], 'g--', label="gyroX")
    ax3.plot(gyro[:, 1], 'b--', label="gyroY")
    ax3.plot(gyro[:, 2], 'c--', label="gyroZ")
    ax3.plot(method1*gyro.max(), 'y', label=name1)
    ax3.plot(method2*gyro.max(), 'm', label=name2)
    ax3.legend(loc=0)

    plt.show()

    return 0


