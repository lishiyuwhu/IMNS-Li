#! /usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2017/12/25 16:09
# @Author  : Shiyu Li
# @Software: PyCharm


# 将对应的cover和stego图像相减

from PIL import Image
import numpy as np

img_c = Image.open('1.pgm')
img_s = Image.open('11.pgm')

data_c = np.array(img_c.getdata())
data_s = np.array(img_s.getdata())

dis = data_c - data_s
dis = (dis+1)*255


img_dis = dis.reshape([256,256])
img = Image.fromarray(img_dis)

np.set_printoptions(threshold=np.nan)
img.save('dis1.pgm')
# if __name__ == '__main__':
#     main()