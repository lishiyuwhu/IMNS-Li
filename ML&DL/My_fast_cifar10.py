#! /usr/bin/python
# -*- coding: utf8 -*-

""" tl.prepro for data augmentation """

import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import *
import numpy as np
import time, os, io
from PIL import Image

sess = tf.InteractiveSession()

X_train, y_train, X_test, y_test = tl.files.load_cifar10_dataset(
                                    shape=(-1, 32, 32, 3), plotable=False)

def model(x, y_, reuse):
    W_init1 = tf.truncated_normal_initializer(stddev=0.0001)
    W_init2 = tf.truncated_normal_initializer(stddev=0.001)
    W_init3 = tf.truncated_normal_initializer(stddev=0.1)
    b_init = tf.constant_initializer(value=0.1)
    with tf.variable_scope("model", reuse=reuse):
        tl.layers.set_name_reuse(reuse)
        net = InputLayer(x, name='input')
        net = Conv2d(net, 32, (5, 5), (2, 2), act=tf.nn.relu,
                    padding='SAME', W_init=W_init1, b_init=None, name='cnn1')
        net = MaxPool2d(net, (3, 3), (2, 2), padding='SAME',name='pool1')

        net = Conv2d(net, 32, (5, 5), (2, 2), act=tf.nn.relu,
                    padding='SAME', W_init=W_init1, b_init=None, name='cnn2')
        net = MeanPool2d(net, (3, 3), (2, 2), padding='SAME',name='pool2')

        net = Conv2d(net, 64, (5, 5), (2, 2), act=tf.nn.relu,
                    padding='SAME',W_init=W_init2, b_init=None, name='cnn3')
        net = MeanPool2d(net, (3, 3), (2, 2), padding='SAME', name='pool3')
        net = FlattenLayer(net, name='flatten')                             # output: (batch_size, 2304)
        net = DenseLayer(net, n_units=64, act=tf.nn.relu,
                    W_init=W_init3, name='d1relu')           # output: (batch_size, 384)
        net = DenseLayer(net, n_units=10, act=tf.identity,
                    W_init=tf.truncated_normal_initializer(stddev=1/192.0),
                    name='output')                                          # output: (batch_size, 10)
        y = net.outputs

        ce = tl.cost.cross_entropy(y, y_, name='cost')
        # L2 for the MLP, without this, the accuracy will be reduced by 15%.
        L2 = 0
        for p in tl.layers.get_variables_with_name('relu/W', True, True):
            L2 += tf.contrib.layers.l2_regularizer(0.004)(p)
        cost = ce + L2

        correct_prediction = tf.equal(tf.argmax(y, 1), y_)
        acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        return net, cost, acc


def distort_fn(x, is_train=False):
    """
    Description
    -----------
    The images are processed as follows:
    .. They are cropped to 24 x 24 pixels, centrally for evaluation or randomly for training.
    .. They are approximately whitened to make the model insensitive to dynamic range.
    For training, we additionally apply a series of random distortions to
    artificially increase the data set size:
    .. Randomly flip the image from left to right.
    .. Randomly distort the image brightness.
    """
    # print('begin',x.shape, np.min(x), np.max(x))
    x = tl.prepro.crop(x, 24, 24, is_random=is_train)
    # print('after crop',x.shape, np.min(x), np.max(x))
    if is_train:
        # x = tl.prepro.zoom(x, zoom_range=(0.9, 1.0), is_random=True)
        # print('after zoom', x.shape, np.min(x), np.max(x))
        x = tl.prepro.flip_axis(x, axis=1, is_random=True)
        # print('after flip',x.shape, np.min(x), np.max(x))
        x = tl.prepro.brightness(x, gamma=0.1, gain=1, is_random=True)
        # print('after brightness',x.shape, np.min(x), np.max(x))
        # tmp = np.max(x)
        # x += np.random.uniform(-20, 20)
        # x /= tmp
    # normalize the image
    x = (x - np.mean(x)) / max(np.std(x), 1e-5) # avoid values divided by 0
    # print('after norm', x.shape, np.min(x), np.max(x), np.mean(x))
    return x

x = tf.placeholder(tf.float32, shape=[None, 32, 32, 3], name='x')
y_ = tf.placeholder(tf.int64, shape=[None, ], name='y_')

# using local response normalization
network, cost, _ = model(x, y_, False)
_, cost_test, acc = model(x, y_, True)

## train
n_epoch = 50000
learning_rate = 0.0001
print_freq = 1
batch_size = 128

train_params = network.all_params
train_op = tf.train.AdamOptimizer(learning_rate, beta1=0.9, beta2=0.999,
    epsilon=1e-08, use_locking=False).minimize(cost, var_list=train_params)

tl.layers.initialize_global_variables(sess)

network.print_params(False)
network.print_layers()

print('   learning_rate: %f' % learning_rate)
print('   batch_size: %d' % batch_size)

for epoch in range(n_epoch):
    start_time = time.time()
    for X_train_a, y_train_a in tl.iterate.minibatches(
                                X_train, y_train, batch_size, shuffle=True):
        sess.run(train_op, feed_dict={x: X_train_a, y_: y_train_a})

    if epoch + 1 == 1 or (epoch + 1) % print_freq == 0:
        print("Epoch %d of %d took %fs" % (epoch + 1, n_epoch, time.time() - start_time))

        test_loss, test_acc, n_batch = 0, 0, 0
        for X_test_a, y_test_a in tl.iterate.minibatches(
                                    X_test, y_test, batch_size, shuffle=False):
            err, ac = sess.run([cost_test, acc], feed_dict={x: X_test_a, y_: y_test_a})
            test_loss += err; test_acc += ac; n_batch += 1
        print("   test loss: %f" % (test_loss/ n_batch))
        print("   test acc: %f" % (test_acc/ n_batch))
        