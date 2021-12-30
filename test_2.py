from cv2 import cv2
import numpy as np
from matplotlib import pyplot as plt

# 灰度化
def rgb2gray(src):
    height = src.shape[0]
    width = src.shape[1]
    red_channel, green_channel, blue_channel = cv2.split(src)
    dst = np.zeros(red_channel.shape, red_channel.dtype)
    for h in range(height):
        for w in range(width):
            dst[h][w] = (red_channel[h][w] * 299 +
                         green_channel[h][w] * 587 +
                         blue_channel[h][w] * 144 +
                         500) / 1000
    # np.savetxt('gray_image.txt', dst, "%d", " ")
    return dst


# 直方图阈值分割
def cacle_hist(src):
    height = src.shape[0]
    width = src.shape[1]
    flat_array = src.ravel()
    plt.hist(flat_array, 256)
    plt.show()
    dst = np.zeros(src.shape, dtype=src.dtype)
    dst.fill(255)
    for h in range(height):
        for w in range(width):
            if src[h][w] <= 125:
                dst[h][w] = 0
            else:
                pass
    return dst


# 读取RGB原图
source_image = cv2.imread(r'img.png')
gray_image = rgb2gray(source_image)
thresh_image = cacle_hist(gray_image)

cv2.imshow('thresh_image', thresh_image)
cv2.waitKey(0)