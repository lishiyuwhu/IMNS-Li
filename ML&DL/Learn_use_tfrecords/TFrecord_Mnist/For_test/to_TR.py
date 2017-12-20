#! /usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2017/12/12 10:40
# @Author  : Shiyu Li
# @Software: PyCharm

import os
import tensorflow as tf
from PIL import Image  # 注意Image,后面会用到
import matplotlib.pyplot as plt
import numpy as np

cwd = os.getcwd() + '\\1\\'
classes = {'zero', 'one'}
writer = tf.python_io.TFRecordWriter("1.tfrecords")  # 要生成的文件

for index, name in enumerate(classes):
    class_path = cwd + name + '\\'
    for img_name in os.listdir(class_path):
        img_path = class_path + img_name

        img = Image.open(img_path)
        img = img.resize((28,28))
        img_raw = img.tobytes()  # 将图片转化为二进制格式
        example = tf.train.Example(features=tf.train.Features(feature={
            "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[index])),
            'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
        }))  # example对象对label和image数据进行封装
        writer.write(example.SerializeToString())  # 序列化为字符串

writer.close()

# if __name__ == '__main__':
#     main()