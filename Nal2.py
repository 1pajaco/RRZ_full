import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

def simple_gaussdx(sigma = 2):
    r = int(np.ceil(3.0 * sigma))
    x = np.arange(-r, r + 1)
    g = -1 / (np.pow(sigma, 3) * np.sqrt(2 * np.pi)) * x * np.exp( - np.pow(x, 2) / (2 * np.pow(sigma, 2)))

    return g / (np.sum(np.abs(g)) / 2) 

def gaussdx_2d(im):
    im = im.astype(np.float32)
    D = simple_gaussdx()[None, :]
    x = cv.filter2D(im, -1, D)
    y = cv.filter2D(im, -1, D.T)
    return x, y

def a():
    im = cv.imread("./src/zmaj.png", cv.IMREAD_GRAYSCALE)
    x, y = gaussdx_2d(im)

    plt.subplot(1, 3, 1)
    plt.imshow(im, cmap='gray')
    plt.title('I')

    plt.subplot(1, 3, 2)
    plt.imshow(x, cmap='gray')
    plt.title('Ix')

    plt.subplot(1, 3, 3)
    plt.imshow(y, cmap='gray')
    plt.title('Iy')

    plt.suptitle("a, odvodi slike")
    plt.show()

def b():
    rez = []
    im = np.zeros((100, 100), dtype=np.float32)
    im[50, 50] = 1.0

    G = cv.getGaussianKernel(51, 9).T
    D = simple_gaussdx(9)[None, :]

    rez.append(im)

    rez.append(cv.filter2D(cv.filter2D(im, -1, G), -1, D.T))
    rez.append(cv.filter2D(cv.filter2D(im, -1, D), -1, G.T))

    rez.append(cv.filter2D(cv.filter2D(im, -1, G), -1, G.T))

    rez.append(cv.filter2D(cv.filter2D(im, -1, G.T), -1, D))
    rez.append(cv.filter2D(cv.filter2D(im, -1, D.T), -1, G))

    for ind, i in enumerate(rez):
        plt.subplot(2, 3, ind+1)
        plt.imshow(i, cmap='gray')
    
    plt.suptitle("b, impulzni signal")
    plt.show()

if __name__ == "__main__":
    pass
    # a(), b()


