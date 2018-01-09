#! /usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2018/1/8 20:17
# @Author  : Shiyu Li
# @Software: PyCharm


import tensorlayer as tl
from tensorlayer.layers import *
import numpy as np
import time

# if __name__ == '__main__':
#     main()


# lrelu
lrelu = lambda x: tl.act.lrelu(x, 0.2)

# prelu
def prelu(_x, scope=None):
    """parametric ReLU activation"""
    with tf.variable_scope(name_or_scope=scope, default_name="prelu"):
        _alpha = tf.get_variable("prelu", shape=_x.get_shape()[-1],
                                 dtype=_x.dtype, initializer=tf.constant_initializer(0.1))
        return tf.maximum(0.0, _x) + _alpha * tf.minimum(0.0, _x)


def PReLU(_x, name=None):
  if name is None:
    name = "alpha"
  _alpha = tf.get_variable(name,
                           shape=_x.get_shape(),
                           initializer=tf.constant_initializer(0.0),
                           dtype=_x.dtype)

  return tf.maximum(_alpha*_x, _x)