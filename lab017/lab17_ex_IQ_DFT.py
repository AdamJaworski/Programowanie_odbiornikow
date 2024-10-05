import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import spectrogram, resample_poly, lfilter, firwin, decimate
import soundfile as sf
import sounddevice as sd

# Color maps for gray printing
cm_plasma = plt.cm.plasma
cm_inferno = plt.cm.inferno
cm = cm_plasma

# Read recorded IQ signal - choose one
# FileName = 'SDRSharp_DABRadio_229069kHz_IQ.wav'; T=1;  demod=0; # DAB Radio signal
# FM Radio signal
#FileName = 'SDRSharp_FMRadio_101600kHz_IQ.wav'
# FileName = 'websdr_recording_2024-10-05T15_12_23Z_648.0kHz.wav'
# T=0
# demod=3

# FileName = 'SDRSharp_NanoSat_146000kHz_IQ.wav'  # Nano Satellite
# T = 0
# demod = 2
# FileName = 'SDRSharp_Airplane_112500kHz_IQ.wav'
# T=5
# demod=3
# VOR airplane
# FileName = 'SDRWeb_Unknown_3671.9kHz.wav';       T=0;  demod=4; # speech from WebSDR

# Get audio info
info = sf.info(FileName)
print(info)
fs = info.samplerate

# Read audio file
if T == 0:
    x, fs = sf.read(FileName, dtype='float32')
else:
    frames = int(T * fs)
    x, fs = sf.read(FileName, start=0, stop=frames, dtype='float32')

Nx = len(x)

# Reconstruct the complex-value IQ data, if necessary add Q=0
if x.ndim == 1:
    x = x + 1j * np.zeros_like(x)
elif x.shape[1] == 2:
    x = x[:, 0] - 1j * x[:, 1]
else:
    x = x[:, 0] + 1j * np.zeros(x.shape[0])

nd = np.arange(0, 2500)
plt.figure(1)
plt.plot(nd, np.real(x[nd]), 'bo-', label='I(n)')
plt.plot(nd, np.imag(x[nd]), 'r*--', label='Q(n)')
plt.xlabel('n')
plt.grid(True)
plt.title('I(n) = (o) BLUE/solid    |    Q(n)= (*) RED/dashed')
plt.legend()
plt.show()

# Parameters - lengths of FFT and STFT, central signal sample
Nc = int(np.floor(Nx / 2))
Nfft = min(2 ** 17, 2 * Nc)
Nstft = 512

# Power Spectral Density (PSD) of the signal
start = int(Nc - Nfft // 2)
end = int(Nc + Nfft // 2)
if start < 0:
    start = 0
if end > Nx:
    end = Nx
n = np.arange(start, end)
df = fs / Nfft
f = df * np.arange(Nfft)
fshift = df * np.arange(-Nfft // 2, Nfft // 2)
w = np.kaiser(Nfft, 10)
X = np.fft.fft(x[n] * w)
P = 2 * np.abs(X) ** 2 / (fs * np.sum(w ** 2))
Pshift = np.fft.fftshift(P)

# Parameters for Short Time Fourier Transform (STFT) of the signal
N = Nstft
df_stft = fs / N
ff = df_stft * np.arange(N)
ffshift = df_stft * np.arange(-N // 2, N // 2)

# Plot PSD and spectrogram for frequencies [0-fs)
plt.figure(2)
plt.subplot(211)
plt.plot(f, 10 * np.log10(np.abs(P)))
plt.xlabel('f (Hz)')
plt.ylabel('(dB/Hz)')
plt.title('PSD for frequencies [0-fs)')
plt.grid(True)
plt.tight_layout()

frequencies, times, Sxx = spectrogram(
    x[n], fs=fs, window=('kaiser', 10), nperseg=Nstft, noverlap=Nstft - Nstft // 4
)
plt.subplot(212)
plt.pcolormesh(times, frequencies, 10 * np.log10(Sxx), shading='gouraud')
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.title('Spectrogram')
plt.colorbar()
plt.tight_layout()
plt.set_cmap(cm)
plt.show()

# Plot PSD and spectrogram for frequencies [-fs/2, fs/2)
plt.figure(3)
plt.subplot(211)
plt.plot(fshift, 10 * np.log10(np.abs(Pshift)))
plt.xlabel('f (Hz)')
plt.ylabel('(dB/Hz)')
plt.title('PSD for frequencies [-fs/2, fs/2)')
plt.grid(True)
plt.tight_layout()

frequencies, times, Sxx = spectrogram(
    x[n],
    fs=fs,
    window=('kaiser', 10),
    nperseg=Nstft,
    noverlap=Nstft - Nstft // 4,
    return_onesided=False,
)
Sxx = np.fft.fftshift(Sxx, axes=0)
frequencies = np.fft.fftshift(np.fft.fftfreq(Nstft, d=1/fs))

plt.subplot(212)
plt.pcolormesh(times, frequencies, 10 * np.log10(Sxx), shading='gouraud')
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.title('Spectrogram with frequencies [-fs/2, fs/2)')
plt.colorbar()
plt.tight_layout()
plt.set_cmap(cm)
plt.show()

# Demodulation
if demod == 1:
    # FM demodulation and mono FM radio decoding
    bwSERV = 200e3
    bwAUDIO = 25e3
    D1 = int(round(fs / bwSERV))
    D2 = int(round(bwSERV / bwAUDIO))
    f0 = -0.59e6
    x = x * np.exp(-1j * 2 * np.pi * f0 / fs * np.arange(len(x)))
    x = resample_poly(x, up=1, down=D1)
    x_demod = np.real(x[1:]) * np.imag(x[:-1]) - np.real(x[:-1]) * np.imag(x[1:])
    x = resample_poly(x_demod, up=1, down=D2)
    x = x / np.max(np.abs(x))
    sd.play(x, int(bwAUDIO))
    sd.wait()

elif demod == 2:
    """
    Tutaj trochę nie ogarniam o co chodzi , zmiana fAudio i f0 empirycznie niczego nie zmienia
    """
    # FM demodulation and NanoSat voice control signal decoding
    fAudio = 24e4
    Down = int(round(fs / fAudio))
    #f0 = 1.98e4
    f0 = 15e5
    x = x * np.exp(-1j * 2 * np.pi * f0 / fs * np.arange(len(x)))
    h = firwin(501, cutoff=12500, fs=fs, pass_zero='lowpass')
    x = lfilter(h, 1.0, x)
    x = x[::Down]
    dt = 1 / fAudio
    x_diff = x[1:] * np.conj(x[:-1])
    x = np.angle(x_diff) / (2 * np.pi * dt)
    x = x / np.max(np.abs(x))
    sd.play(x, int(fAudio))
    sd.wait()

elif demod == 3:
    # AM demodulation and VOR voice control signal decoding
    #fc = 2.9285e5
    #fc = 3.55e5
    #fam = 10000
    fc = 648e3
    fam = 0.1e5
    f1 = fc - fam / 2
    f2 = fc + fam / 2
    df = 500
    h = firwin(501, [f1 + df, f2 - df], fs=fs, pass_zero=False)
    x = lfilter(h, 1.0, x)
    x = np.abs(x)
    D = int(round(fs / fam))
    x = decimate(x, D)
    x = x / np.max(np.abs(x))
    sd.play(x, int(fam))
    sd.wait()

elif demod == 4:
    # Speech from WebSDR
    x = x / np.max(np.abs(x))
    sd.play(x, int(fs))
    sd.wait()
