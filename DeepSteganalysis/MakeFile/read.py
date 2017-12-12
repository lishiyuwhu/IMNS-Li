#! /usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2017/12/7 17:28
# @Author  : Shiyu Li
# @Software: PyCharm


# if __name__ == '__main__':
#     main()
import tensorflow as tf
import numpy as np
from PIL import Image

fileNameQue = tf.train.string_input_producer(["test.cifar10"])
reader = tf.TFRecordReader()
key, value = reader.read(fileNameQue)
features = tf.parse_single_example(value, features={'label': tf.FixedLenFeature([], tf.int64),
                                                    'img_raw': tf.FixedLenFeature([], tf.string), })

img = tf.decode_raw(features["img_raw"], tf.uint8)
label = tf.cast(features["label"], tf.int32)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    for i in range(2):
        imgArr = sess.run(img)
        print(imgArr)

    coord.request_stop()
    coord.join(threads)
