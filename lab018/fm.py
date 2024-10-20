import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
from scipy.signal import spectrogram
from scipy.fft import fft
import sounddevice as sd

# Example of FM modulation and demodulation

# Modulating signal m(t)
x, fs = sf.read('GOODBYE.WAV')  # Read speech from file
if x.ndim > 1:
    x = x[:, 0]  # Take only one channel
Nx = len(x)                      # Number of signal samples
dt = 1 / fs
t = dt * np.arange(Nx)           # Time vector
df = 1 / (Nx * dt)
f = df * np.arange(Nx)           # Frequency vector

# x = np.cos(2 * np.pi * 2 * t)  # Alternative modulating signal

plt.figure()
plt.plot(t, x)
plt.xlabel('t [s]')
plt.grid()
plt.title('x(t)')
plt.show()

# Spectrogram of x(t)
f_spec, t_spec, Sxx = spectrogram(x, fs=fs, window='hann', nperseg=256, noverlap=192, nfft=512)
plt.figure()
plt.pcolormesh(t_spec, f_spec, 10 * np.log10(Sxx), shading='gouraud')
plt.ylabel('Frequency [kHz]')
plt.xlabel('Time [s]')
plt.title('STFT of x(t)')
plt.colorbar(label='Intensity [dB]')
plt.show()

# FM modulation
fc = 0                           # Carrier frequency: 0 or 4000 Hz
BW = 0.9 * fs                    # Available bandwidth
fmax = 3500                      # Maximum modulating frequency
df_calc = (BW / (2 * fmax) - 1) * fmax  # Calculated frequency modulation depth
df = 1000                        # Arbitrarily chosen frequency modulation depth

y = np.exp(1j * 2 * np.pi * (fc * t + df * np.cumsum(x) * dt))  # Signal modulated in frequency
Y = np.abs(fft(y) / Nx)                                          # Its DFT spectrum

plt.figure()
plt.plot(f, Y)
plt.grid()
plt.xlabel('f [Hz]')
plt.title('|Y(f)|')
plt.show()

plt.figure()
plt.plot(f, 20 * np.log10(Y))
plt.grid()
plt.xlabel('f [Hz]')
plt.title('|Y(f)| [dB]')
plt.show()

# Spectrogram of y(t)
f_y, t_y, Syy = spectrogram(y, fs=fs, window='hann', nperseg=256, noverlap=192, nfft=512)
plt.figure()
plt.pcolormesh(t_y, f_y, 10 * np.log10(np.abs(Syy)), shading='gouraud')
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [s]')
plt.title('STFT of y(t)')
plt.colorbar(label='Intensity [dB]')
plt.show()

# FM demodulation methods
ang = np.unwrap(np.angle(y))
fi1 = (1 / (2 * np.pi)) * (ang[1:] - ang[:-1]) / dt              # M1
fi2 = (1 / (2 * np.pi)) * np.angle(y[1:] * np.conj(y[:-1])) / dt  # M2
fi3 = (1 / (2 * np.pi)) * np.angle(y[2:] * np.conj(y[:-2])) / (2 * dt)
fi3 = np.append(fi3, 0)                                           # M3
fi4 = (1 / (2 * np.pi)) * (
    (np.real(y[1:-1]) * (np.imag(y[2:]) - np.imag(y[:-2])) -
     np.imag(y[1:-1]) * (np.real(y[2:]) - np.real(y[:-2])))
) / (2 * dt)
fi4 = np.append(fi4, 0)                                           # M4
fi5 = (1 / (2 * np.pi)) * (
    np.real(y[:-1]) * np.imag(y[1:]) - np.imag(y[:-1]) * np.real(y[1:])
) / dt                                                            # M5

nn = np.arange(len(fi1))
plt.figure()
plt.plot(nn, fi1, 'r', nn, fi2, 'g', nn, fi3, 'b', nn, fi4, 'k', nn, fi5, 'm')
plt.title('Calculated angles')
plt.xlabel('Sample index')
plt.legend(['φ₁', 'φ₂', 'φ₃', 'φ₄', 'φ₅'])
plt.show()

# Recover the modulating signal
xest = (fi2 - fc) / df  # Recovered modulating signal
xest = xest[:-1]
x = x[1:-1]
t = t[1:-1]

plt.figure()
plt.plot(t, x, 'r-', label='Original')
plt.plot(t, xest, 'b-', label='Demodulated')
plt.xlabel('t [s]')
plt.title('Original and Demodulated Signal')
plt.grid()
plt.legend()
plt.show()

# Spectrogram of xest(t)
f_xest, t_xest, Sxx_est = spectrogram(xest, fs=fs, window='hann', nperseg=256, noverlap=192, nfft=512)
plt.figure()
plt.pcolormesh(t_xest, f_xest, 10 * np.log10(Sxx_est), shading='gouraud')
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [s]')
plt.title('STFT of xest(t)')
plt.colorbar(label='Intensity [dB]')
plt.show()

# ERROR after frequency MOD & DEMOD
ERROR_SIGNAL = np.max(np.abs(x - xest))
print('ERROR_SIGNAL =', ERROR_SIGNAL)  # FM demodulation error

# Playing the original signal
sd.play(x / np.max(np.abs(x)), fs)
sd.wait()

# Playing the demodulated signal
sd.play(xest / np.max(np.abs(xest)), fs)
sd.wait()
