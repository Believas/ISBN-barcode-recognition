from cv2 import cv2
import numpy as np


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
    np.savetxt('gray_image.txt', dst, "%d", " ")
    return dst


# 读取RGB原图
source_image = cv2.imread(r'img.png')
# source_image = cv2.resize(source_image, (600,400))
gray_image = rgb2gray(source_image)
cv2.imshow('gray_image', gray_image)
cv2.waitKey(0)