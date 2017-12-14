#! /usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2017/12/10 15:12
# @Author  : Shiyu Li
# @Software: PyCharm


import tensorflow as tf
import numpy as np
from PIL import Image

import os


def TRencode(file):
    cwd = os.getcwd()

    root = cwd + '/' + file

    name = "./minstTR." + file
    TFwriter = tf.python_io.TFRecordWriter(name)

    for className in os.listdir(root):
        label = int(className[0])
        classPath = root + "/" + className + "/"
        for parent, dirnames, filenames in os.walk(classPath):
            for filename in filenames:
                imgPath = classPath + "/" + filename
                print(imgPath)
                img = Image.open(imgPath)
                print(img.size, img.mode)
                imgRaw = img.tobytes()
                example = tf.train.Example(features=tf.train.Features(feature={
                    "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
                    "img_raw": tf.train.Feature(bytes_list=tf.train.BytesList(value=[imgRaw]))
                }))
                TFwriter.write(example.SerializeToString())

    TFwriter.close()


if __name__ == '__main__':
    TRencode("pngtest")
    TRencode("pngtrain")
