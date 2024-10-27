# lab19_ex_am.py
# Example of AM modulation and demodulation in Python

import numpy as np
import scipy.signal as signal
from scipy.io import wavfile

# Parameters
sig = 1     # Signal type: 1=speech, 2=cos, 3=cos+cos, 4=SFM
mod = 5     # AM type: 1=DSB-LC, 2=DSB-SC, 3=SSB-U=USB, 4=SSB-L=LSB, 5=DSB-SC-CX
demod = 1   # Demodulator type: 1=Hilbert transform, 2=Quadrature
nosynch = 0 # Carrier synchronization in frequency down-conversion (0/1)
disturb = 0 # Disturbance - second AM service using frequency (2*fc) (0/1)
noise = 0   # Presence of noise (0/1)
fc = 50000  # AM carrier frequency jak jest na 5/4 to min błąd dla am
dA = 0.5    # AM modulation depth for DSB-LC
Nwin = 256
Nover = Nwin - 32
Nfft = 2 * Nwin   # For STFT plot

# Modulating signal x(t)
fss, x = wavfile.read('speech11025.wav')  # Read mono audio from file, fs=11025 Hz
x = x.astype(float)  # Ensure data type is float
K = 5
x = signal.resample_poly(x, K, 1)  # Upsample K-times for frequency-UP
fs = K * fss
N = len(x)  # Number of signal samples
dt = 1 / fs
t = dt * np.arange(N)  # Time vector
df = fs / Nfft
f = df * np.arange(-Nfft//2, Nfft//2)  # Frequencies for STFT display

# Alternative signals for tests
if sig == 2:
    x = np.cos(2 * np.pi * 2000 * t)  # 1x cos
elif sig == 3:
    x = np.cos(2 * np.pi * 2000 * t) + np.cos(2 * np.pi * 3000 * t)  # 2x cos
elif sig == 4:
    x = np.cos(2 * np.pi * (2000 * t + 1000 / (2 * np.pi * 5) * np.sin(2 * np.pi * 5 * t)))  # SFM

# Create base-band signal for AM modulation of the carrier: x(t) --> a(t)
if mod == 1:
    a = (1 + dA * x)  # DSB-LC
elif mod == 2:
    a = x  # DSB-SC
elif mod == 3:
    a = x + 1j * np.imag(signal.hilbert(x))  # SSB-U = USB
elif mod == 4:
    a = x - 1j * np.imag(signal.hilbert(x))  # SSB-L = LSB
elif mod == 5:
    x = x + 1j * np.sin(2 * np.pi * 4000 * t)
    a = x  # DSB-SC-CMPLX

# Carrier AM modulation - frequency-up conversion: y(t) = a(t)*c(t)
c = np.exp(1j * 2 * np.pi * fc * t)
y = np.real(a) * np.real(c) - np.imag(a) * np.imag(c)  # y = np.real(a * c); the same

# Possible service using frequency (2*fc) (optional)
if disturb == 1:
    y += np.real(a) * np.cos(2 * np.pi * (2 * fc) * t) - np.imag(a) * np.sin(2 * np.pi * (2 * fc) * t)

# Additive noise
if noise == 1:
    y += 0.025 * np.random.randn(N)

# Possible lack of synchronization between frequency up-shifter and down-shifter
if nosynch == 1:
    c = np.exp(1j * (2 * np.pi * (fc + 100) * t + np.pi / 4))  # Carrier used for freq down-conversion

# Carrier AM demodulation - frequency-down conversion
if demod == 1:
    yH = signal.hilbert(y)  # Hilbert filter - analytic signal
    a1 = yH * np.conj(c)    # Frequency-down conversion to base-band
elif demod == 2:
    a1 = 2 * y * np.real(c) - 2j * y * np.imag(c)  # Quadrature demodulator

# Low-pass filter design and filtering
M = 100
h = signal.firwin(2*M+1, cutoff=(fss/2)/(fs/2), window=('kaiser', 12))  # LP filter design
a2 = np.convolve(a1, h)
a2 = a2[M:-M]  # Filtering

# Recovering x(t) from a(t)
if mod == 1:
    xdem = (np.abs(a2) - 1) / dA  # DSB-LC
elif mod in [2, 3, 4]:
    xdem = np.real(a2)  # DSB-SC, SSB-U, SSB-L
elif mod == 5:
    xdem = a2  # DSB-SC-CMPLX

# Optional trimming
n = slice(499, N - 500)
t = t[n]
x = x[n]
xdem = xdem[n]

# ERROR of demodulation
ERROR_Demod_SIGNAL = np.max(np.abs(x - xdem))
print('ERROR_Demod_SIGNAL =', ERROR_Demod_SIGNAL)
ERROR_Demod_SPECTRUM = np.max(np.abs(np.fft.fft(x) - np.fft.fft(xdem))) / N
print('ERROR_Demod_SPECTRUM =', ERROR_Demod_SPECTRUM)
