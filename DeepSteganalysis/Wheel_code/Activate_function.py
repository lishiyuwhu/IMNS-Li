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

#by Tensorlayer
class PReluLayer(Layer):
    """
    The :class:`PReluLayer` class is Parametric Rectified Linear layer.

    Parameters
    ----------
    x : A `Tensor` with type `float`, `double`, `int32`, `int64`, `uint8`,
        `int16`, or `int8`.
    channel_shared : `bool`. Single weight is shared by all channels
    a_init : alpha initializer, default zero constant.
        The initializer for initializing the alphas.
    a_init_args : dictionary
        The arguments for the weights initializer.
    name : A name for this activation op (optional).

    References
    -----------
    - `Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification <http://arxiv.org/pdf/1502.01852v1.pdf>`_
    """
    def __init__(
        self,
        layer = None,
        channel_shared = False,
        a_init = tf.constant_initializer(value=0.0),
        a_init_args = {},
        # restore = True,
        name="prelu_layer"
    ):
        Layer.__init__(self, name=name)
        self.inputs = layer.outputs
        print("  [TL] PReluLayer %s: channel_shared:%s" % (self.name, channel_shared))
        if channel_shared:
            w_shape = (1,)
        else:
            w_shape = int(self.inputs.get_shape()[-1])

        # with tf.name_scope(name) as scope:
        with tf.variable_scope(name) as vs:
            alphas = tf.get_variable(name='alphas', shape=w_shape, initializer=a_init, dtype=D_TYPE, **a_init_args )
            try:  ## TF 1.0
                self.outputs = tf.nn.relu(self.inputs) + tf.multiply(alphas, (self.inputs - tf.abs(self.inputs))) * 0.5
            except: ## TF 0.12
                self.outputs = tf.nn.relu(self.inputs) + tf.mul(alphas, (self.inputs - tf.abs(self.inputs))) * 0.5


        self.all_layers = list(layer.all_layers)
        self.all_params = list(layer.all_params)
        self.all_drop = dict(layer.all_drop)

        self.all_layers.extend( [self.outputs] )
        self.all_params.extend( [alphas] )





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