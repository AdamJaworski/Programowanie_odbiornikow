from scipy.fft import fft, ifft
import numpy as np
import utilities


# FM modulated signal
Nx = 2000                 # number of signal samples
fs = 2000                 # sampling frequency [Hz]
fc = 400                  # carrier frequency [Hz]
fm = 2                    # FM modulation frequency [Hz]
df = 50                   # FM modulation depth
dt = 1/fs                 # time step
t = dt * np.arange(Nx)    # time vector
xm = np.sin(2 * np.pi * fm * t)  # modulating signal

N = 8
k_A = 0.5
a_k = np.random.rand(N)
phi_k = 2 * np.pi * np.random.rand(N)
c_k = np.array([(N/2 + k + 1) * np.exp(1j * k * np.pi / N) for k in range(-N//2, N//2)])
# # AM-SC
# x = xm
#
# # AM-LC
# x = (1 + (xm * k_A))
#
# # FM modulated signal
# x = np.exp(1j * (2 * np.pi * 0 * t + 2 * np.pi * df / fs * np.cumsum(xm)))
#
# # R-OFDM
# N = len(a_k)
# omega_0 = 2 * np.pi / (t[-1] - t[0])
# x = np.zeros_like(t)
# for k in range(1, N):
#     x += a_k[k] * np.cos(k * omega_0 * t + phi_k[k])
# C-OFDM
N = len(c_k)
omega_0 = 2 * np.pi / (t[-1] - t[0])  # Fundamental frequency
x = np.zeros_like(t, dtype=complex)
for k in range(-N // 2, N // 2):
    x += c_k[k] * np.exp(1j * k * omega_0 * t)

utilities.plot_signal(x, fs)
utilities.plot_spectrum(x, fs)

# Frequency UP and REAL part
c = np.cos(2 * np.pi * fc * t)
s = np.sin(2 * np.pi * fc * t)
xUp = x * (c + 1j * s)      # Frequency UP - carrier AM

utilities.plot_spectrum(xUp, fs)

xUpReal = np.real(xUp)      # REAL part

utilities.plot_spectrum(xUpReal, fs)

# Frequency DOWN
xDownCos = 2 * xUpReal * c
xDownSin = -2 * xUpReal * s
xDown = xDownCos + 1j * xDownSin

utilities.plot_spectrum(xDown, fs)

# Ideal LowPass Filter in frequency domain
df = fs / Nx
K = int(np.floor(fc / df))
XDownFilt = fft(xDown)       # FFT
XDownFilt[K+1:Nx-K] = 0      # zero unwanted DFT bins
xDownFilt = ifft(XDownFilt)  # IFFT

utilities.plot_spectrum(xDownFilt, fs)

# ERROR after frequency UP & DOWN
error_signal = np.max(np.abs(x - xDownFilt))
print(f"ERROR SIGNAL: {error_signal}")
