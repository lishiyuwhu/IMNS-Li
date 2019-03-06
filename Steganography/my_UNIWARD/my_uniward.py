#! /usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2019/3/5 9:47
# @Author  : Shiyu Li
# @Software: PyCharm


import numpy as np
import cv2


def dis(img):
    img2 = img
    if img.dtype != 'uint8':
        img2 = img2.astype(np.uint8)

    cv2.namedWindow("Image")
    cv2.imshow("Image", img2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

class Myuniward:
    def __init__(self):
        self.img = None
