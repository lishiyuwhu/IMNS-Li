#! /usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2017/12/1 17:31
# @Author  : Shiyu Li
# @Software: PyCharm

'''
import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import *
import numpy as np
import time

def read_and_decode_without_distortion(filename):

    filename_queue = tf.train.string_input_producer([filename])
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    feature = tf.parse_single_example(serialized_example,
                                      features={
                                          'label':tf.FixedLenFeature([], tf.int64),
                                          'img_raw':tf.FixedLenFeature([], tf.string),
                                      })
    img = tf.decode_raw(feature['img_raw'], tf.float32)
    img = tf.reshape(img, [256,256,1])
    #img = tf.cast(img, tf.float32) * (1./255) - 0.5
    label = tf.cast(feature['label'], tf.int32)

    return img, label

x , y =read_and_decode_without_distortion('test.tfrecords')

print(type(x))
'''
# if __name__ == '__main__':
#     main()
'''

import tensorflow as tf
import numpy as np

i=0
for serialized_example in tf.python_io.tf_record_iterator("test.cifar10"):
    example = tf.train.Example()
    example.ParseFromString(serialized_example)

    image = example.features.feature['image'].bytes_list.value
    label = example.features.feature['label'].int64_list.value
    # 可以做一些预处理之类的
    break
'''



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
    # You can do more image distortion here for training data
    img = tf.decode_raw(features['img_raw'], tf.float32)
    img = tf.reshape(img, [32, 32, 3])
    label = tf.cast(features['label'], tf.int32)
    return img, label

# Example to visualize data
img, label = read_and_decode("train.cifar10")
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
