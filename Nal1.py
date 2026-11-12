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
# helpers
def simple_convolution(I, k):
    k_r = k[::-1]
    N = len(k_r) // 2
    arr = np.array([])

    for i in range(N, len(I) - N):
        arr = np.append(arr, np.sum(I[i-N : i+N+1] * k_r))

    return arr

def simple_gauss(sigma=2):
    r = int(np.ceil(3.0 * sigma))
    x = np.arange(-r, r + 1)
    g = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-(np.pow(x,2)) / (2*np.pow(sigma,2)))

    return g / np.sum(g)

def simple_gaussdx(sigma = 2):
    r = int(np.ceil(3.0 * sigma))
    x = np.arange(-r, r + 1)
    g = -1 / (np.pow(sigma, 3) * np.sqrt(2 * np.pi)) * x * np.exp( - np.pow(x, 2) / (2 * np.pow(sigma, 2)))

    return g / (np.sum(np.abs(g)) / 2) 

def gauss_filter(im, sigma = 2):
    kernel = simple_gauss(sigma)
    kernel = kernel[..., None]
    kernel = kernel @ kernel.T

    rez = cv.filter2D(im, -1, kernel)

    return rez

def median_filter_2d(im, kernel_s):
    n = len(im)

    padded = np.pad(im, kernel_s // 2, mode='constant', constant_values=0)
    rez = np.zeros_like(im)

    for i in range(n):
        for j in range(len(im[i])):
            sorted = np.sort(padded[i : i + kernel_s, j : j + kernel_s].flatten())
            rez[i, j] = sorted[len(sorted) // 2]

    return rez

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
    plt.suptitle("b, prikaz simple konvolucije")

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

    x = np.arange(len(signal))
    plt.plot(x, signal, label='signal')

    x = np.arange(len(conv))
    plt.plot(x, conv, label='Conv brez same')

    x = np.arange(len(same))
    plt.plot(x, same, label='Conv z same')

    plt.suptitle("c, prikaz uporabe same")

    plt.legend()
    plt.show()

# d https://computervisiononline.com/explains/gaussian-filter/
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
    plt.xticks([])
    plt.yticks([])
    plt.subplot(1, 2, 2)
    plt.plot(x, kernel, 'o-', )
    plt.title('kernel')
    plt.xticks([])
    plt.yticks([])

    plt.suptitle(f'e, Prikaz kernel in simple_gauss sigma 2\nVsota {np.sum(g)}')
    plt.show()

    sigma = [0.5, 1, 2, 3, 4]
    plt.suptitle('e, 2D Gauss matrike')
    for i in sigma:
        g = simple_gauss(i)
        r = len(g) // 2
        x = np.arange(-r, r + 1)

        plt.plot(x, g, 'o-', label=f'sigma = {i}')
        plt.title(f'Sigma: {sigma}')

    plt.suptitle("e, 1D prikaz gaussovih kernalov z razlicno sigmo")
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

    plt.suptitle("e, 2D prikaz gaussovih kernalov z razlicno sigmo")
    plt.show()

# f; f(), first kernel will make the signal more rounded whilte
# the second one will 

def f():
    signal = np.interp(np.arange(125),
                   [0, 15, 35, 45, 52, 75, 78, 100, 120, 124],
                   [1, 1, 0, 0, 0.6, 0.6, 0, 0, 0.5, 0.5])

    rez = [signal]
    titles = ["Original", "Gaussian smoothing", "Gaussian derivative"]

    kernel = simple_gauss()
    rez.append(np.convolve(signal, kernel, mode='valid'))


    # kernel = simple_gaussdx()
    kernel = np.gradient(kernel)
    rez.append(np.convolve(signal, kernel, mode='valid'))

    for ind, i in enumerate(rez):
        plt.subplot(1, 3, ind+1)
        plt.plot(i)
        plt.title(titles[ind])

    plt.show()

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
    plt.suptitle("g, Prikaz asociativnosti konvolucije")
    plt.show() 
    

def h():
    im, rez = [], []
    tim = cv.imread("./src/cameraman_gauss_less.png", cv.IMREAD_GRAYSCALE)
    im.append(tim.astype(np.float32))
    tim = cv.imread("./src/cameraman_sp.png", cv.IMREAD_GRAYSCALE)
    im.append(tim.astype(np.float32))

    #seperabilnost
    kernel = simple_gauss(2)
    kernel = kernel[..., None]
    kernel_2d = kernel @ kernel.T

    for i in im:
        tim1 = cv.filter2D(i, -1, kernel)
        tim2 = cv.filter2D(tim1, -1, kernel.T)
        tim3 = cv.filter2D(i, -1, kernel_2d)

        rez.extend([i, tim1, tim2, tim3])

    titles = ["Original", "1. pass", "2. pass", "2D pass"]
    plt.suptitle("h, Primerjava seperabilnosti gausovega jedra.")

    for ind, i in enumerate(rez):
        plt.subplot(2, 4, ind + 1)
        plt.imshow(i, cmap='gray')
        plt.title(titles[ind % 4])

    plt.show()

    # Najbolje odstrani Gaussov sum

def i():
    im = cv.imread("./src/fox.jpg")
    rez = [im[..., ::-1].astype(np.float32)]

    kernel = \
    [[0, 0, 0],
     [0, 2, 0],
     [0, 0, 0]] - 1/9 * np.ones((3,3), dtype=np.float32)

    tim = cv.filter2D(rez[0], -1, kernel)
    rez.append(tim)
    titles = ['Original', 'Ostrenje pass 1']

    for i in range(4):
        tim = cv.filter2D(tim, -1, kernel)
        rez.append(tim)
        titles.append(f"Ostrenje pass {i+2}")

    print(titles)

    for ind, i in enumerate(rez):
        plt.subplot(2, 3, ind+1)
        plt.imshow(np.clip(i, 0, 255).astype(np.uint8))
        plt.title(titles[ind])

        plt.xticks([])
        plt.yticks([])

    plt.suptitle("i, Prikaz stopnje ostrenja")
    plt.show()

    # povzroci da se slika pokvari

def median_filter_1d(signal, kernel_s):
    N = kernel_s // 2
    odd = kernel_s % 2 != 0
    rez = []

    signal_p = np.pad(signal, N, mode='edge')
    
    for i in range(len(signal)):
        if odd:
            sorted = np.sort(signal_p[i : i + kernel_s])
            rez.append(sorted[N])
        else:
            sorted = np.sort(signal_p[i : i + kernel_s])
            rez.append((sorted[N - 1] + sorted[N]) / 2)
        # rez.append(np.sort(signal[i - N : i+N+1])[N])
    return rez

def j():
    x = np.concatenate((np.zeros(14), np.ones(11), np.zeros(15)))
    # pokvarjen signal
    xc = np.copy(x)
    xc[11] = 5
    xc[18] = 5

    rez = median_filter_1d(xc, 5)
    gauss = cv.getGaussianKernel(5, 2).flatten()
    gauss = np.convolve(xc, gauss, mode='same')

    plt.plot(xc, label='signal')
    plt.plot(rez, label='median')
    plt.plot(gauss, label='gauss')

    plt.suptitle("j, 1D prikaz filtriranja z guass in median kernel")
    plt.legend()
    plt.show()

def k():
    im, rez = [], []

    tim = cv.imread("./src/cameraman_gauss_less.png", cv.IMREAD_GRAYSCALE)
    im.append(tim.astype(np.float32))
    tim = cv.imread("./src/cameraman_sp_less.png", cv.IMREAD_GRAYSCALE)
    im.append(tim.astype(np.float32))

    titles = ['Gaussian noise', 'Gauss filtered', 'Median filtered', 'Salt-and-pepper noise', 'Gauss filtered', 'Median filtered']

    for i in im:
        gauss = gauss_filter(i)
        med = median_filter_2d(i, 3)

        rez.extend([i, gauss, med])


    for ind, i in enumerate(rez):
        plt.subplot(2, 3, ind + 1)
        plt.imshow(i, cmap='gray')

        plt.title(titles[ind])
        plt.xticks([])
        plt.yticks([])
    plt.suptitle("k, Prikaz filtriranja z Gaussovim in median kernel")
    plt.show()

if __name__ == "__main__":
    pass
    # a(), b(), c(), d(), f(), e(), g(), h(), i(), j(), k()





