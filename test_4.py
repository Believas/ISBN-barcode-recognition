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


# 先切割一下没有必要的前景，只保留数字块
def find_aera(src):
    height = src.shape[0]
    width = src.shape[1]
    start = -1
    end = -1
    for h in range(height):
        for w in range(width):
            if src[h][w] == 0:
                break
            elif w == width - 1:
                start = h + 5
            else:
                pass
        if start != -1:
            break
    for h in range(height - 1, 0, -1):
        for w in range(width):
            if src[h][w] == 0:
                break
            elif w == width - 1:
                end = h - 5
            else:
                pass
        if end != -1:
            break
    ROI = src[start:end, :]
    contours, hierarchy = cv2.findContours(ROI, mode=cv2.RETR_TREE,method=cv2.CHAIN_APPROX_SIMPLE)
    # for i in contours:
    #     length = cv2.arcLength(i, True)
    #     print(length)
    cv2.drawContours(ROI, contours,-1,color=255,thickness=1)
    ROI_inv = cv2.bitwise_not(ROI)
    retval, labels, stats, centroids = cv2.connectedComponentsWithStats(ROI_inv, labels=None, stats=None, centroids=None, connectivity=8, ltype=None)


    print(stats[:,0])
    stats_ = np.argsort(stats[:,0])
    print(stats_)
    for i in stats_:
        print('The rectangular box coordinates are (', stats[i][0], stats[i][1], ')\t(',
              stats[i][0] + stats[i][2], stats[i][1] + stats[i][3], ')\t\t',
              'The area are ', stats[i][4], '\t\t', 'The centroid', centroids[i], '\t\t', 'The lenth are', 2 * (stats[i][2] * stats[i][3]))







    plt.imshow(labels)
    plt.show()
    return ROI


# 读取RGB原图
source_image = cv2.imread(r'img.png')
gray_image = rgb2gray(source_image)
thresh_image = cacle_hist(gray_image)
ROI = find_aera(thresh_image)
cv2.imshow('ROI', ROI)
cv2.waitKey(0)



