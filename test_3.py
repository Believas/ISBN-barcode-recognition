from cv2 import cv2
import numpy as np
import math
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


# 生成高斯核
def gausskernel(size):
    sigma = 1.0
    gausskernel = np.zeros((size, size), np.float32)
    for i in range(size):
        for j in range(size):
            norm = math.pow(i - 1, 2) + pow(j - 1, 2)
            gausskernel[i, j] = math.exp(-norm / (2 * math.pow(sigma, 2)))  # 求高斯卷积
    sum = np.sum(gausskernel)  # 求和
    kernel = gausskernel / sum  # 归一化
    return kernel


# 高斯滤波
def gausssion_filter(src):
    height = src.shape[0]
    width = src.shape[1]
    dst = np.zeros((height, width), np.uint8)
    kernel = gausskernel(3)  # 计算高斯卷积核
    for h in range(1, height - 1):
        for w in range(1, width - 1):
            sum = 0
            for k in range(-1, 2):
                for l in range(-1, 2):
                    sum += src[h + k, w + l] * kernel[k + 1, l + 1]  # 高斯滤波
            dst[h, w] = sum
    return dst


#sobel算子进行边缘检测
def sobel_edge(src):
    image = gausssion_filter(src)
    # sobel 1 算子模板 2 图片卷积 3 阈值判决
    # [1  2  1           [1  0  -1
    # 0  0  0            2  0  -2
    # -1 -2 -1]         1   0  -1]
    height = image.shape[0]
    width = image.shape[1]
    # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    dst = np.zeros((height, width, 1), np.uint8)
    for i in range(0, height - 2):
        for j in range(0, width - 2):
            # 计算x y方向的梯度
            gy = image[i, j] * 1 + image[i, j + 1] * 2 + image[i, j + 2] * 1 - image[i + 2, j] * 1 - image[
                i + 2, j + 1] * 2 - image[i + 2, j + 2] * 1
            gx = image[i, j] * 1 + image[i + 1, j] * 2 + image[i + 2, j] * 1 - image[i, j + 2] * 1 - image[
                i + 1, j + 2] * 2 - image[i + 2, j + 2] * 1
            grad = math.sqrt(gx * gx + gy * gy)
            if grad > 50:
                dst[i, j] = 255
            else:
                dst[i, j] = 0
    return dst


# 读取RGB原图
source_image = cv2.imread(r'img.png')
gray_image = rgb2gray(source_image)
filter_image = gausssion_filter(gray_image)
edge_image = sobel_edge(gray_image)

cv2.imshow('gaussion_filter', filter_image)
cv2.imshow('edge_sobel', edge_image)
cv2.waitKey(0)


