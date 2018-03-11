#! /usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2017/12/12 20:08
# @Author  : Shiyu Li
# @Software: PyCharm

import tensorflow as tf
import numpy as np
from PIL import Image
import cv2

def dis(img):
    img2 = img
    if img.dtype != 'uint8':
        img2 = img2.astype(np.uint8)

    cv2.namedWindow("Image")
    cv2.imshow("Image", img2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

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
    img = tf.decode_raw(features['img_raw'], tf.float32)
    img = tf.reshape(img, [256, 256, 1])
    # img = tf.cast(img, tf.float32) # if you want to use tfrecords as input.
    label = tf.cast(features['label'], tf.int32)
    return img, label

# visualize data
img, label = read_and_decode("own_CroppedBossBase-1.0-256x256.tfrecords")
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
    dis(val[0])

    coord.request_stop()
    coord.join(threads)
    sess.close()