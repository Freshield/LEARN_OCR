# coding=utf-8
"""
@Author: Freshield
@Contact: yangyufresh@163.com
@File: a1_test_example.py
@Time: 2020-07-02 17:28
@Last_update: 2020-07-02 17:28
@Desc: None
@==============================================@
@      _____             _   _     _   _       @
@     |   __|___ ___ ___| |_|_|___| |_| |      @
@     |   __|  _| -_|_ -|   | | -_| | . |      @
@     |__|  |_| |___|___|_|_|_|___|_|___|      @
@                                    Freshield @
@==============================================@
"""
import cv2
import numpy as np


def verticae_projection(thresh1):
    h, w = thresh1.shape
    a = [0 for z in range(0, w)]
    for j in range(0, w):
        for i in range(0, h):
            if thresh1[i, j] == 0:
                a[j] += 1
                thresh1[i, j] = 255

    for j in range(0, w):
        for i in range(h - a[j], h):
            thresh1[i, j] = 0

    roi_list = list()
    start_index = 0
    end_index = 0
    in_block = False
    for i in range(0, w):
        if in_block == False and a[i] != 0:
            in_block = True
            start_index = i
        elif a[i] == 0 and in_block:
            end_index = i
            in_block = False
            print()
            print('here')
            print(start_index)
            print(end_index)
            roiImg = thresh1[0: h, start_index: end_index+1]
            cv2.imshow('img', roiImg)
            cv2.waitKey()
            roi_list.append(roiImg)

    return roi_list


img = cv2.imread('data/idcard1.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, thresh1 = cv2.threshold(img, 130, 255, cv2.THRESH_BINARY)
print(np.unique(thresh1))
roi_list = verticae_projection(thresh1)
for thresh1 in roi_list:
    print(np.unique(thresh1))
    cv2.imshow('img', thresh1)
    cv2.waitKey()