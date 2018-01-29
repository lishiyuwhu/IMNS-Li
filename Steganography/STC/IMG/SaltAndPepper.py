#! /usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2018/1/22 21:01
# @Author  : Shiyu Li
# @Software: PyCharm

import cv2
import numpy as np

def SaltAndPepper(src,percetage):
    NoiseImg=src
    NoiseNum=int(percetage*src.shape[0]*src.shape[1])
    for i in range(NoiseNum):
        xj=np.random.random_integers(0,src.shape[0]-1)
        xi=np.random.random_integers(0,src.shape[1]-1)
        if np.random.random_integers(0,1)==0:
            NoiseImg[xj,xi,0] = 255
            NoiseImg[xj,xi,1] = 255
            NoiseImg[xj,xi,2] = 255
        else:
            NoiseImg[xj,xi,0] = 25
            NoiseImg[xj,xi,1] = 20
            NoiseImg[xj,xi,2] = 20
    return NoiseImg

if __name__ == '__main__':
    img = cv2.imread('stego_.jpg')
    NoiseImg = SaltAndPepper(img, 0)
    cv2.imwrite('stego.jpg', NoiseImg, [cv2.IMWRITE_JPEG_QUALITY, 100])