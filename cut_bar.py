from cv2 import cv2
import numpy as np
from matplotlib import pyplot as plt
import os
import multiprocessing as mp
from cv2 import cv2
import math


# 霍夫线变换
def lines_detector_hough(edge, ThetaDim=None, DistStep=None, threshold=None, halfThetaWindowSize=2,
                         halfDistWindowSize=None):
    '''
    :param edge: 经过边缘检测得到的二值图
    :param ThetaDim: hough空间中theta轴的刻度数量(将[0,pi)均分为多少份),反应theta轴的粒度,越大粒度越细
    :param DistStep: hough空间中dist轴的划分粒度,即dist轴的最小单位长度
    :param threshold: 投票表决认定存在直线的起始阈值
    :return: 返回检测出的所有直线的参数(theta,dist)
    @author: believas
    '''
    imgsize = edge.shape
    if ThetaDim == None:
        ThetaDim = 90
    if DistStep == None:
        DistStep = 1
    MaxDist = np.sqrt(imgsize[0] ** 2 + imgsize[1] ** 2)
    DistDim = int(np.ceil(MaxDist / DistStep))

    if halfDistWindowSize == None:
        halfDistWindowSize = int(DistDim / 50)
    accumulator = np.zeros((ThetaDim, DistDim))  # theta的范围是[0,pi). 在这里将[0,pi)进行了线性映射.类似的,也对Dist轴进行了线性映射

    sinTheta = [np.sin(t * np.pi / ThetaDim) for t in range(ThetaDim)]
    cosTheta = [np.cos(t * np.pi / ThetaDim) for t in range(ThetaDim)]

    for i in range(imgsize[0]):
        for j in range(imgsize[1]):
            if not edge[i, j] == 0:
                for k in range(ThetaDim):
                    accumulator[k][int(round((i * cosTheta[k] + j * sinTheta[k]) * DistDim / MaxDist))] += 1

    M = accumulator.max()

    if threshold == None:
        threshold = int(M * 2.3875 / 4.5)
    result = np.array(np.where(accumulator > threshold))  # 阈值化
    temp = [[], []]
    for i in range(result.shape[1]):
        eight_neiborhood = accumulator[
                           max(0, result[0, i] - halfThetaWindowSize + 1):min(result[0, i] + halfThetaWindowSize,
                                                                              accumulator.shape[0]),
                           max(0, result[1, i] - halfDistWindowSize + 1):min(result[1, i] + halfDistWindowSize,
                                                                             accumulator.shape[1])]
        if (accumulator[result[0, i], result[1, i]] >= eight_neiborhood).all():
            temp[0].append(result[0, i])
            temp[1].append(result[1, i])

    result = np.array(temp)  # 非极大值抑制

    result = result.astype(np.float64)
    result[0] = result[0] * np.pi / ThetaDim
    result[1] = result[1] * MaxDist / DistDim
    draw = drawLines(result, edge)
    # plt.imshow(draw)
    # plt.show()

    return result

# 绘制霍夫线
def drawLines(lines, edge, color=(255, 0, 0), err=3):
    if len(edge.shape) == 2:
        result = np.dstack((edge, edge, edge))
    else:
        result = edge
    Cos = np.cos(lines[0])
    Sin = np.sin(lines[0])

    for i in range(edge.shape[0]):
        for j in range(edge.shape[1]):
            e = np.abs(lines[1] - i * Cos - j * Sin)
            if (e < err).any():
                result[i, j] = color

    return result

#旋转校正
def rotate_transfer(src, output_image):
    # src一定是边缘二值图
    hough_lines = lines_detector_hough(src)
    lines = []
    saves = []
    if hough_lines.shape[1] <= 1:
        return output_image
    else:
        for i in range(hough_lines.shape[1]):
            if hough_lines[0][i] < 0.56 * np.pi and hough_lines[0][i] > 0.44 * np.pi:
                saves.append(hough_lines[0][i])
        M = cv2.getRotationMatrix2D((600, 400), 90 - np.mean(saves) * 180 / np.pi, 0.98)
        # print('ratate:', 90 - np.mean(saves) * 180 / np.pi)
        dst = cv2.warpAffine(output_image, M, (600, 400), cv2.BORDER_CONSTANT, borderValue=255)
    return dst


# 水平投影
def getHProjection(src):
    image = src.copy()
    (h, w) = image.shape  # 返回高和宽
    # 初始化一个跟图像高一样长度的数组，用于记录每一行的黑点个数
    a = [0 for z in range(0, h)]
    for j in range(0, h):  # 遍历每一行
        for i in range(0, w):  # 遍历每一列
            if image[j, i] == 0:  # 判断该点是否为黑点，0代表黑点
                a[j] += 1  # 该行的计数器加一
                image[j, i] = 255  # 将其改为白点，即等于255
    for j in range(0, h):  # 遍历每一行
        for i in range(0, a[j]):  # 从该行应该变黑的最左边的点开始向最右边的点设置黑点
            image[j, i] = 0  # 设置黑点


    dst = image[:, :-110]
    # plt.imshow(image, cmap=plt.gray())
    # plt.show()
    etval, labels, stats, centroids = cv2.connectedComponentsWithStats(dst, connectivity=8)
    return dst, stats


def find_baseline(array):
    list_a = array[:, 4].tolist()
    del (list_a[0])
    max_list = max(list_a)  # 返回最大值
    max_index = list_a.index(max(list_a))  # 最大值的索引
    shrot_distance = 1000
    for i in range(1, array.shape[0]):
        if array[i][1] < array[max_index + 1][1] and array[max_index + 1][1] - array[i][1] < shrot_distance:
            shrot_distance = array[max_index + 1][1] - array[i][1]
            index = i
    return index


def process(source_path, target_path):
    try:

        source_image = cv2.imread(source_path)
        source_img = cv2.resize(source_image, (600, 400))

        filter_image = cv2.GaussianBlur(source_img, (7, 7), 0)
        gray_image = cv2.cvtColor(filter_image, cv2.COLOR_BGR2GRAY)

        adaptive_image = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 13, 6)
        adaptive_image_ = cv2.morphologyEx(adaptive_image, cv2.MORPH_CLOSE, (3, 3))

        edge_image = cv2.Canny(gray_image, 50, 150)

        rotate_image = rotate_transfer(edge_image, adaptive_image_)
        rotate_image_inv = cv2.bitwise_not(rotate_image)

        Hprojection_image, stats_projection = getHProjection(rotate_image_inv)

        index = find_baseline(stats_projection)

        ROI = rotate_image_inv[stats_projection[index][1] - 5:stats_projection[index][1] + stats_projection[index][3] + 7,:]

        cv2.imwrite(target_path, ROI)

    except Exception as e:
        print(e)


if __name__ == '__main__':
    path = 'data_test/'
    obj_path = 'write_dir/'
    img_paths = os.listdir(path)
    result = 0
    i_ = 0

    # 多进程运算， processes的数值最大为cpu的最大进程数量
    pool = mp.Pool(processes=3)
    for img in img_paths:
        i_ += 1
        abs_path = os.path.join(path, img)
        img_name = img.split('.')[0]
        pool.apply_async(process, (abs_path, obj_path + img_name + '.jpg'))
    pool.close()
    pool.join()
