#! /usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2018/1/22 15:50
# @Author  : Shiyu Li
# @Software: PyCharm

import Jsteg
import cv2
import numpy as np


def dct(img):
    img = np.float32(img) / 255.0
    return cv2.dct(img) * 255

def idct(img):
    img = np.float32(img) / 255.0
    return cv2.idct(img) * 255

def display(img):
    cv2.namedWindow("Image") 
    cv2.imshow("Image", img)
    cv2.waitKey (0)
    cv2.destroyAllWindows()  


if __name__ == '__main__':    
    imgfile = "stego_pgm.pgm"
    img = cv2.imread(imgfile, 0)
    jsteg = Jsteg.Jsteg()
    
    img = img.flatten().tolist()
    
    sequence_after_dct = dct(img)
    jsteg.set_sequence_after_dct(sequence_after_dct)
    stego_info = jsteg.read()
    print(stego_info)

'''
    jsteg.set_sequence_after_dct(sequence_after_dct)
    info1 = [0, 1, 0, 1, 1, 0, 1, 0]
    jsteg.write(info1)
    sequence_after_dct2 = jsteg.get_sequence_after_dct()
    # 读
    jsteg.set_sequence_after_dct(sequence_after_dct2)
    info2 = jsteg.read()
    print(info2)
'''