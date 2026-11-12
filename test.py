import numpy as np
import matplotlib.pyplot as plt

signal = np.concatenate((
    np.ones(15),
    np.linspace(1, 0, 20),
    np.zeros(10),
    np.linspace(0, 0.6, 7),
    np.ones(23) * 0.6,
    np.zeros(22),
    np.linspace(0, 0.5, 20),
    np.ones(4) * 0.5
))

sigma = 3
g = simple_gauss(sigma)
d = simple_gaussdx(sigma)
r = len(g) // 2

padded = np.pad(signal, r, mode='edge')
smooth = np.convolve(padded, g, mode='valid')
deriv = np.convolve(padded, d, mode='valid')

x = np.arange(len(signal))

plt.subplot(3, 1, 1)
plt.plot(x, signal)
plt.subplot(3, 1, 2)
plt.plot(x, signal, alpha=0.4)
plt.plot(x, smooth)
plt.subplot(3, 1, 3)
plt.plot(x, deriv)
plt.axhline(0, color='gray')
plt.show()