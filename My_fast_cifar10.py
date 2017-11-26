#! /usr/bin/python
# -*- coding: utf8 -*-

'''
Created on Nov 20, 2017


@author: Li

'''


# Use fast cifar10 model http://www.cnblogs.com/neopenx/p/4480701.html
# without BN-layers

import tensorflow as tf 
import tensorlayer as tl
from tensorlayer.layers import *
import numpy as np 
import time, os, io 
from PIL import Image

sess = tf.InteractiveSession()

X_train, y_train, X_test, y_test = tl.files.load_cifar10_dataset(
    shape=(-1,32,32,3), plotable=False)

def Fast_model_without_BN(x, y_, reuse, is_train):

    # initialization
    W_init1 = tf.truncated_normal_initializer(stddev=0.0001)
    W_init2 = tf.truncated_normal_initializer(stddev=0.001)
    W_init3 = tf.truncated_normal_initializer(stddev=0.1)
    b_init = tf.constant_initializer(value=0.1)


    with tf.variable_scope("model", reuse=reuse):
        tl.layers.set_name_reuse(reuse)
        net = InputLayer(x, 'input1')
        net = Conv2d(net, 32, (5,5), (2,2), padding='SAME',
                     W_init=W_init1, b_init=None, name='cnn1')
        net = MaxPool2d(net, (3,3), (2,2), padding='SAME', name='max_pooling1')
        net = Conv2d(net, 32, (5,5), (2,2), padding='SAME',
                     W_init=W_init2, b_init=None, name='cnn2')
        net = MeanPool2d(net, (3,3), (2,2), padding='SAME', name='avg_pooling2')
        net = Conv2d(net, 64, (5,5), (2,2), padding='SAME',
                     W_init=W_init2, b_init=None, name='cnn3')
        net = MeanPool2d(net, (3,3), (2,2), padding='SAME', name='avg_pooling3')
        net = FlattenLayer(net, name='flatten')
        net = DenseLayer(net, n_units=64, act=tf.nn.relu,
                         W_init=W_init3, b_init=b_init, name='full_connected')
        net = DenseLayer(net, n_units=10, act=tf.nn.relu,
                         W_init=W_init3, b_init=b_init, name='output1')
        y = net.outputs

        ce = tl.cost.cross_entropy(y, y_, name='cost')
        L2 = 0
        for p in tl.layers.get_layers_with_name(net, 'full_connected', True):
            L2 += tf.contrib.layers.l2_regularizer(0.004)(p)
        cost = ce +L2

        correct_prediction = tf.equal(tf.argmax(y, 1), y_)
        acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        return net, cost, acc


x = tf.placeholder(tf.float32, shape=[None, 32, 32, 3], name='x')
y_ = tf.placeholder(tf.int64, shape=[None, ], name='y_')

network, cost, _ = Fast_model_without_BN(x, y_, False, is_train=True)
_, cost_test, acc = Fast_model_without_BN(x, y_, True, is_train=False)

# train params
n_epoch = 5000
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
            test_loss += err
            test_acc += ac
            n_batch += 1
        print("   test loss: %f" % (test_loss / n_batch))
        print("   test acc: %f" % (test_acc / n_batch))
