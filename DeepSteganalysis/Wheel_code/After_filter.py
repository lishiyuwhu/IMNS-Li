# ! /usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2018/1/11 10:31
# @Author  : Shiyu Li
# @Software: PyCharm

import tensorflow as tf
import tensorlayer as tl
from PIL import Image
import numpy as np

img_path = '1.pgm'
img_size = [256,256]




img = Image.open(img_path)
data = np.array(img.getdata())
data = data.astype(np.float32)
data = data.reshape([1, img_size[0], img_size[1], 1])
sess = tf.InteractiveSession()
# define placeholder
x = tf.placeholder(tf.float32, shape=[None, img_size[0], img_size[1], 1], name='x')

# define the network
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
# define cost function and metric.
y = net.outputs
img_filtered = tf.cast(y, dtype=tf.uint8)
# initialize all variables in the session
tl.layers.initialize_global_variables(sess)
y_img = sess.run([y], feed_dict={x:data})
y_img = np.array(y_img)
img_filter = Image.fromarray(y_img.reshape(img_size[0], img_size[1]))


