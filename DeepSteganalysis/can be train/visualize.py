#! /usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2017/12/8 14:22
# @Author  : Shiyu Li
# @Software: PyCharm


import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import *
import numpy as np
import time
from PIL import Image
import os
import io


def read_and_decode(filename):
    """ Return tensor to read from TFRecord """
    filename_queue = tf.train.string_input_producer([filename])
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'label': tf.FixedLenFeature([], tf.int64),
                                           'img_raw' : tf.FixedLenFeature([], tf.string),
                                       })
    img = tf.decode_raw(features['img_raw'], tf.uint8)
    print(type(img))
    print(tf.size(img))
    img = tf.reshape(img, [256, 256, 1])
    label = tf.cast(features['label'], tf.int32)
    return img, label

# Example to visualize data
img, label = read_and_decode("boossbase.test")
img_batch, label_batch = tf.train.shuffle_batch([img, label],
                                                batch_size=4,
                                                capacity=50000,
                                                min_after_dequeue=10000,
                                                num_threads=1)
print("img_batch   : %s" % img_batch._shape)
print("label_batch : %s" % label_batch._shape)

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    for i in range(3):  # number of mini-batch (step)
        print("Step %d" % i)
        val, l = sess.run([img_batch, label_batch])
        # exit()
        print(val.shape, l)
        tl.visualize.images2d(val, second=1, saveable=False, name='batch'+str(i), dtype=np.uint8, fig_idx=2020121)

    coord.request_stop()
    coord.join(threads)
    sess.close()