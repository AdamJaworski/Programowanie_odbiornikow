import numpy as np
from scipy.signal import sosfilt, butter
import matplotlib.pyplot as plt
from utilities import plot_spectrum

# Constants
ENCODING_WORD = "testowy caig znakow"
carrier_freq = 5000  # Carrier frequency in Hz
bit_rate = 1000      # Bit rate in bits per second
fs = 10 * carrier_freq  # Sampling frequency
bit_duration = 1 / bit_rate
samples_per_bit = int(bit_duration * fs)

# Generate binary data
binary_data = ''.join(format(ord(c), '08b') for c in ENCODING_WORD)
binary_data_list = [int(b) for b in binary_data]
bit_levels = np.array([1 if b == 1 else -1 for b in binary_data_list])

# Time vector
total_samples = samples_per_bit * len(binary_data_list)
t = np.arange(total_samples) / fs

# Message signal m(t)
m = np.repeat(bit_levels, samples_per_bit)

# AM Modulation
s = (1 + m) * np.cos(2 * np.pi * carrier_freq * t)

# Add AWGN noise
def add_awgn_noise(signal, snr_db=0.1):
    signal_power = np.mean(signal ** 2)
    snr_linear = 10 ** (snr_db / 10)
    noise_power = signal_power / snr_linear
    noise = np.random.normal(0, np.sqrt(noise_power), len(signal))
    return signal + noise

noisy_signal = add_awgn_noise(s)

# Receiver
# 1. Bandpass Filter
nyquist = 0.5 * fs
low = (carrier_freq - bit_rate) / nyquist
high = (carrier_freq + bit_rate) / nyquist
sos_butter_bandpass = butter(2, [low, high], btype='bandpass', output='sos')
filtered_signal = sosfilt(sos_butter_bandpass, noisy_signal)

# 2. Downconversion
# c_ = np.cos(2 * np.pi * carrier_freq * t)
# s_ = np.sin(2 * np.pi * carrier_freq * t)
#
# x_down = filtered_signal * (c_ + 1j * s_)
x_down = filtered_signal * np.cos(2 * np.pi * carrier_freq * t)

plot_spectrum(x_down, fs)

# 3. Low-Pass Filter
lowcut = 1.5 * bit_rate
low = lowcut / nyquist
sos_butter_lowpass = butter(2, low, btype='low', output='sos')
x_baseband = sosfilt(sos_butter_lowpass, x_down)

# Envelope Detection
envelope = np.abs(x_baseband)

# Sampling at bit centers
sample_points = np.arange(samples_per_bit // 2, len(envelope), samples_per_bit)
sampled_values = envelope[sample_points]

# Thresholding
threshold = (np.max(envelope) + np.min(envelope)) / 2
recovered_bits = ['1' if val > threshold else '0' for val in sampled_values]
recovered_data = ''.join(recovered_bits)

# Convert binary data back to string
chars = [chr(int(recovered_data[i:i+8], 2)) for i in range(0, len(recovered_data), 8)]
print("Decoded String:", ''.join(chars))

# Plotting
plt.figure(figsize=(12, 8))

plt.subplot(4, 1, 1)
plt.title('Original Message Signal')
plt.plot(t, m)
plt.xlim(0, bit_duration * 30)
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')

plt.subplot(4, 1, 2)
plt.title('Original Message Signal')
plt.plot(t, s)
plt.xlim(0, bit_duration * 30)
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')

plt.subplot(4, 1, 3)
plt.title('Modulated Signal with Noise')
plt.plot(t, noisy_signal)
plt.xlim(0, bit_duration * 30)
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')

plt.subplot(4, 1, 4)
plt.title('Recovered Envelope')
plt.plot(t, envelope)
plt.xlim(0, bit_duration * 30)
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')

plt.tight_layout()
plt.show()
