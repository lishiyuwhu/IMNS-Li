# -*- coding: utf-8 -*-
"""
Created on Mon Mar 12 13:16:27 2018

@author: Li
@Software: Spyder
"""

import JSteg
import cv2
import numpy as np
import math

a = JSteg.JSteg()
a.set_img('7.pgm')
a.write('42_85p85.pgm')

img_raw = cv2.imread('7.pgm',0).astype(np.float32)

img_stego = a.encode_img
img_norm = img_raw
img_norm[96:,:] = img_stego[96:,:]


# PSNR
dif = img_norm - img_stego
dif = dif.flatten()
dif = dif.astype(np.uint8)

s = np.sum(dif**2)
MSE = s/(256*256)

PSNR = 10*math.log(10,(255**2/MSE))
print(PSNR)


def psmr(im1,im2):
    diff = numpy.abs(im1 - im2)
    diff = diff.flatten()

    rmse = sum(x*x for x in diff)
    psnr = 20*numpy.log10(255/rmse)
    return psnr