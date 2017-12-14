#! /usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2017/12/12 20:08
# @Author  : Shiyu Li
# @Software: PyCharm

import tensorflow as tf
import numpy as np
from PIL import Image


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
    img = tf.reshape(img, [512, 512, 1])
    # img = tf.cast(img, tf.float32) # if you want to use tfrecords as input.
    label = tf.cast(features['label'], tf.int32)
    return img, label

# visualize data
img, label = read_and_decode("testBossBase-1.01-hugo-alpha=0.4.tfrecords")
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

    val, l = sess.run([img_batch, label_batch])
    print(type(val))
    show_img = Image.fromarray(np.squeeze(val[0], axis=(2,)))
    show_img.show()

    coord.request_stop()
    coord.join(threads)
    sess.close()