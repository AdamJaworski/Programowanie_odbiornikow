import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq


def plot_signal(signal, fs):
    """
    Plots the time-domain signal.

    Parameters:
    - signal: Input signal (1D numpy array)
    - fs: Sampling frequency (in Hz)
    """
    Nx = len(signal)  # Number of samples
    t = np.arange(Nx) / fs  # Time vector based on sampling frequency

    plt.figure(figsize=(10, 4))
    plt.plot(t, signal, label="Signal")
    plt.title('Time-Domain Signal')
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude')
    plt.grid(True)
    plt.legend()
    plt.show()


def plot_spectrum(signal, fs):
    """
    Plots the frequency spectrum of the signal, including negative frequencies.

    Parameters:
    - signal: Input signal (1D numpy array)
    - fs: Sampling frequency (in Hz)
    """
    Nx = len(signal)  # Number of samples
    signal_fft = fft(signal)  # Compute FFT
    freq = fftfreq(Nx, 1 / fs)  # Frequency bins (including negative frequencies)

    # Magnitude of FFT
    signal_magnitude = np.abs(signal_fft)

    plt.figure(figsize=(10, 4))
    plt.plot(freq, signal_magnitude, label="Spectrum")
    plt.title('Frequency Spectrum (with Negative Frequencies)')
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Magnitude')
    plt.grid(True)
    plt.legend()
    plt.show()
