#! /usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2017/12/12 10:47
# @Author  : Shiyu Li
# @Software: PyCharm

import os
import tensorflow as tf
from PIL import Image  # 注意Image,后面会用到
import matplotlib.pyplot as plt
import numpy as np
import tensorlayer as tl

filename_queue = tf.train.string_input_producer(["1.tfrecords"])
reader = tf.TFRecordReader()
_, serialized_example = reader.read(filename_queue)
features = tf.parse_single_example(serialized_example,
                                   features={
                                       'label': tf.FixedLenFeature([], tf.int64),
                                       'img_raw': tf.FixedLenFeature([], tf.string),
                                   })
image = tf.decode_raw(features['img_raw'], tf.uint8)
label = tf.cast(features['label'], tf.int32)
cwd = os.getcwd()

img_batch, label_batch = tf.train.shuffle_batch([image, label],
                                                batch_size=4,
                                                capacity=50000,
                                                min_after_dequeue=10000,
                                                num_threads=1)


with tf.Session() as sess:
    init_op = tf.initialize_all_variables()
    sess.run(init_op)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    for i in range(10):
        example, l = sess.run([img_batch, label_batch])
        print(i)
        print(l)
        print(np.shape(example))

    coord.request_stop()
    coord.join(threads)

# if __name__ == '__main__':
#     main()
