import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

signal = np.array([
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
    5.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 5.0, 
    1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
])

kernel = np.array(
    [
        0.0022181959, 0.0087731348, 0.027023158, 0.064825185, 
        0.12110939,   0.17621312,   0.19967563,  0.17621312, 0.12110939,
        0.064825185,  0.027023158,  0.0087731348, 0.0022181959
    ])

# a.)
def a():
    I = np.array([0, 1, 1, 1, 0, 0.7, 0.5, 0.2, 0, 0, 1, 0])
    ko = np.array([0.5, 1, 0.3])

    ko_inverted = np.array([0.3, 1, 0.5])

    result = np.array([
        0, 0.5, 1.5, 1.8, 1.3, 0.65, 0.95, 0.81, 0.35, 0.06, 0.5, 1, 0.3, 0
    ])

    target = np.convolve(I, ko)
    print(f'a.) Same: {np.allclose(target, result)}')

# b.)
def b():
    def simple_convolution(I, k):
        k_r = k[::-1]
        N = len(k) // 2
        arr = np.array([])

        for i in range(N, len(I) - N):
            arr = np.append(arr, np.sum(I[i-N : i+N+1] * k_r))

        return arr

    target = np.convolve(signal, kernel, mode='valid')
    result = simple_convolution(signal, kernel)

    print(f'b.) Same: {np.allclose(result, target)}')
    # pomembno ker zelimo da vrednosti niso skalirane in imamo normalizirano vrednost kot rezultat
    # to je Gauss jedro z sigma vrednostjo 2
    print(f'Sum of the kernal: {np.sum(kernel)}, skoraj 1')

    x = np.arange(len(signal))
    plt.plot(x, signal, 'o-', label='signal')

    x = np.arange(len(kernel))
    plt.plot(x, kernel, 'o-', label='kernel')

    N = len(kernel) // 2
    x = np.arange(N, len(signal) - N)
    plt.plot(x, result, 'o-', label='rezultat')

    plt.legend()
    plt.show()

# c.)
def c():
    # truncata konvolucijo da output velikost je enak input velikosti
    print(f'c.) Convolve s same: ')

    conv = np.convolve(signal, kernel)
    same = np.convolve(signal, kernel, mode='same')

    print(f'Size of convolve: {conv.size},')
    print(f'Size of same convolve: {same.size},')
    print(f'Size of signal: {signal.size},')

    x = np.arange(len(conv))
    plt.plot(x, conv, label='Conv brez same')

    x = np.arange(len(same))
    plt.plot(x, same, label='Conv z same')

    x = np.arange(len(signal))
    plt.plot(x, signal, label='signal')

    plt.legend()
    plt.show()

# d https://computervisiononline.com/explains/gaussian-filter/
def simple_gauss(sigma):
    r = int(np.ceil(3.0 * sigma))
    x = np.arange(-r, r + 1)
    g = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-(np.pow(x,2)) / (2*np.pow(sigma,2)))

    return g / np.sum(g)

def d():
    sigma = 2
    rez = simple_gauss(sigma)

    target = cv.getGaussianKernel(len(rez), sigma)
    target = np.reshape(target, -1)

    print(f'd.) Same: {np.allclose(target, rez)}')

# e
def e():
    # 2 * np.ceil(3 * sigma) + 1 ~ st. vrednosti
    sigma = 2

    g = simple_gauss(sigma)
    r = len(g) // 2
    x = np.arange(-r, r + 1)

    plt.subplot(1, 2, 1)
    plt.plot(x, g, 'o-')
    plt.title('simple_gauss')
    plt.subplot(1, 2, 2)
    plt.plot(x, kernel, 'o-', )
    plt.title('kernel')

    plt.suptitle(f'Vsota {np.sum(g)}')
    plt.show()

    sigma = [0.5, 1, 2, 3, 4]
    plt.suptitle('2D Gauss matrike')
    for i in sigma:
        g = simple_gauss(i)
        r = len(g) // 2
        x = np.arange(-r, r + 1)

        plt.plot(x, g, 'o-', label=f'sigma = {i}')
        plt.title(f'Sigma: {sigma}')

    plt.legend()
    plt.grid(True)
    plt.show()

    for ind, i in enumerate(sigma):
        g = simple_gauss(i)
        g = g[..., None]
        gxy = g @ g.T

        plt.subplot(2, 3, ind + 1)
        plt.imshow(gxy)
        plt.title(f'Sigma = {i}')
    plt.show()

# f; f(), first kernel will make the signal more rounded whilte
# the second one will 

# g
def g():
    k1 = simple_gauss(2)
    k2 = np.asarray([0.1, 0.6, 0.4])
    k3 = np.convolve(k1, k2)

    rez = []
    rez.append(np.convolve(signal, k1))
    rez[0] = np.convolve(rez[0], k2)

    rez.append(np.convolve(signal, k2))
    rez[1] = np.convolve(rez[1], k1)

    rez.append(np.convolve(signal, k3))

    print(f'g.) Same: {np.allclose(rez[0], rez[1])}, {np.allclose(rez[1], rez[2])}')

    titles = ["k1 * signal -> k2 * signal", "k2 * signal -> k1 * signal", "k1 * k2 -> k3 * signal"]

    plt.subplot(1, 4, 1)
    plt.plot(np.arange(len(signal)), signal, 'o-')
    for ind, i in enumerate(rez):
        plt.subplot(1, 4, ind+2)
        plt.title(titles[ind])
        plt.plot(np.arange(len(i)), i, 'o-')

    plt.show() 
    
# a(), b(), c(), d(), e(), g()

def gauss_filter():
    # soo do we take in an image and a filter? where do i
    # get the specs? how what why dfq am i doing here

    print("whatever")

g()
