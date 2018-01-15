#! /usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2018/1/15 14:52
# @Author  : Shiyu Li
# @Software: PyCharm

import tensorflow as tf
import tensorlayer as tl
from PIL import Image
import numpy as np

sess = tf.InteractiveSession()
x = tf.placeholder(tf.float32, shape=[None, 256,256,1])
F0 = np.array([[-1, 2, -2, 2, -1],
               [2, -6, 8, -6, 2],
               [-2, 8, -12, 8, -2],
               [2, -6, 8, -6, 2],
               [-1, 2, -2, 2, -1]], dtype=np.float32)
F0 = F0 / 12.
# assign numpy array to constant_initalizer and pass to get_variable
high_pass_filter = tf.constant_initializer(value=F0, dtype=tf.float32)
net = tl.layers.InputLayer(x, name='input')
net = tl.layers.Conv2d(net, 1, (5, 5), (1, 1), act=tf.identity, padding='SAME', W_init=high_pass_filter,
                       name='HighPass')
# net1 = tl.layers.DenseLayer(inputs, 800, act=tf.nn.relu, name='relu1_1')
# net2 = tl.layers.DenseLayer(inputs, 300, act=tf.nn.relu, name='relu2_1')
# net = tl.layers.ConcatLayer([net1, net2], 1, name ='concat_layer')

tl.layers.initialize_global_variables(sess)
net.all_params[0].eval()
tl.visualize.CNN2d(net.all_params[0].eval(), second=10, saveable=True, name='cnn1_mnist', fig_idx=2012)
# if __name__ == '__main__':
#     main()

