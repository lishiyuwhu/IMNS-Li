#! /usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2018/1/9 11:02
# @Author  : Shiyu Li
# @Software: PyCharm

from tensorlayer.layers import *

with tf.variable_scope('global_average_pooling'):
    print('xxxxxxxxxxxxx', h)
    gap_filter = net.create_variable('filter', shape=(1, 1, 128, 10))
    h = tf.nn.conv2d(h, filter=gap_filter, strides=[1, 1, 1, 1], padding='SAME')
    print('before global avg:', h)
    h = tf.nn.avg_pool(h, ksize=[1, 4, 4, 1], strides=[1, 1, 1, 1], padding='VALID')
    print(h)
    h = tf.reduce_mean(h, axis=[1, 2])
    net.layers.append(h)

# if __name__ == '__main__':
#     main()