import numpy as np
import matplotlib.pyplot as plt

# Generate a sine wave signal
fs = 200
x = np.arange(0, 1, 1/fs)
y = np.sin(2 * np.pi * x * 10) + np.sin(2 * np.pi * x * 2)

# Compute the Fourier transform
f = np.fft.fft(y)
normalize = len(y)/2
# Get the sample frequencies
freq = np.fft.fftfreq(len(y), d=1/fs)
# Set up the figure and subplots
fig, axs = plt.subplots(1, 2, figsize=(12, 4))

# Plot the time-domain signals
axs[0].plot(x, y, label="sin10t+sin2t")
axs[0].plot(x, np.sin(2 * np.pi * x * 10), label="sin10t", alpha=0.3)
axs[0].plot(x, np.sin(2 * np.pi * x * 2), label="sin2t", alpha=0.3)
axs[0].set_xlabel("Czas (s)")
axs[0].set_ylabel("Amplituda")
axs[0].grid(True)
axs[0].legend()

# Plot the power spectrum
axs[1].plot(freq, abs(f) ** 2)
axs[1].set_xlim(0, 12)
axs[1].set_xlabel("Częstość (Hz)")
axs[1].set_ylabel("Moc")
axs[1].grid(True)

# Show the plot
plt.tight_layout()
plt.show()