import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

from Nal1 import simple_gauss, simple_gaussdx

im = cv.imread("./src/museum.jpg", cv.IMREAD_GRAYSCALE)

#a
def gradient_magnitude(im):

    im = im.astype(np.float32) / 255.0

    g = simple_gauss(1)
    g_x = g[None, :]
    g_y = g_x.T

    gdx = simple_gaussdx(1)
    gdx_x = gdx[None, :]
    gdx_y = gdx_x.T

    Ix = cv.filter2D(cv.filter2D(im, -1, g_y), -1, gdx_x)
    Iy = cv.filter2D(cv.filter2D(im, -1, gdx_y), -1, g_x)

    m = np.sqrt(Ix**2 + Iy**2)
    ang = np.arctan2(Iy, Ix)

    return Ix, Iy, m, ang 

def edges_simple(im, prag = 0.05):
    _, _, m, _ = gradient_magnitude(im)

    return m > prag

def canny():
    print("yep lol")

def a():
    rez = gradient_magnitude(im)

    titles = ['$I_x$', '$I_y$', '$I_{mag}$', '$I_{dir}$']

    plt.subplot(2, 3, 1)
    plt.imshow(im, cmap="gray")
    plt.title('$I$')

    for ind, i in enumerate(rez):
        plt.subplot(2, 3, ind+2)
        plt.imshow(i, cmap="gray")
        plt.title(titles[ind])
    plt.subplot(2, 3, 6)

    m_norm = rez[2] / np.max(rez[2]) 
    plt.imshow(rez[3], cmap='hsv', alpha=m_norm)
    plt.gca().set_facecolor('black')
    plt.title('$I_{dir}$ (HSV)')

    plt.suptitle("a, lokalne spremebe")

    plt.show()

def b():
    vals = [0.05, 0.1, 0.15, 0.2, 0.25, 0.5]

    for ind, val in enumerate(vals):
        plt.subplot(2, 3, ind+1)
        plt.imshow(edges_simple(im, val), cmap='gray')
        plt.title(f"prag {val}")

    plt.suptitle("b, upragovanje magnitude")

    plt.show()

def c():
    cap = cv.VideoCapture("./src/test.mp4")

    while True:
        ret, frame = cap.read()

        cv.imshow("frame", frame)

        frame = cv.resize(frame, None, fx=0.05, fy=0.05)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    pass
    # a(), b() 
    c()