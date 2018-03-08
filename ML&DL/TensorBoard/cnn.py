#! /usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2018/2/27 15:15
# @Author  : Shiyu Li
# @Software: PyCharm

import numpy as np
import tensorlayer as tl
import tensorflow as tf
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import time

# train
batch_size = 128
n_epoch = 20
learning_rate = 0.0001
print_freq = 1

X_train, y_train, _, _, X_test, y_test = \
    tl.files.load_mnist_dataset(shape=(-1, 28, 28, 1))
X_train = np.asarray(X_train, dtype=np.float32)
y_train = np.asarray(y_train, dtype=np.int32)
X_test = np.asarray(X_test, dtype=np.float32)
y_test = np.asarray(y_test, dtype=np.int32)
sess = tf.InteractiveSession()

x = tf.placeholder(tf.float32, shape=[batch_size, 28, 28, 1])   # [batch_size, height, width, channels]
y_ = tf.placeholder(tf.int64, shape=[batch_size,])
network = tl.layers.InputLayer(x, name='input')
network = tl.layers.Conv2d(network, 32, (5, 5), (1, 1),
        act=tf.nn.relu, padding='SAME', name='cnn1')
network = tl.layers.MaxPool2d(network, (2, 2), (2, 2),
        padding='SAME', name='pool1')
network = tl.layers.Conv2d(network, 32, (3, 3), (1, 1),
        act=tf.nn.relu, padding='SAME', name='cnn2')
network = tl.layers.MaxPool2d(network, (2, 2), (2, 2),
        padding='SAME', name='pool2')
network = tl.layers.FlattenLayer(network, name='flatten')
network = tl.layers.DenseLayer(network, 10, act=tf.identity, name='output')
y = network.outputs
cost = tl.cost.cross_entropy(y, y_, 'cost')
correct_prediction = tf.equal(tf.argmax(y, 1), y_)
acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

train_params = network.all_params
train_op = tf.train.AdamOptimizer(learning_rate, beta1=0.9, beta2=0.999,
    epsilon=1e-08, use_locking=False).minimize(cost, var_list=train_params)

tl.layers.initialize_global_variables(sess)
network.print_params()
network.print_layers()


for epoch in range(n_epoch):
    start_time = time.time()
    for X_train_a, y_train_a in tl.iterate.minibatches(
                                X_train, y_train, batch_size, shuffle=True):
        feed_dict = {x: X_train_a, y_: y_train_a}
        feed_dict.update( network.all_drop )        # enable noise layers
        sess.run(train_op, feed_dict=feed_dict)

    if epoch + 1 == 1 or (epoch + 1) % print_freq == 0:
        print("Epoch %d of %d took %fs" % (epoch + 1, n_epoch, time.time() - start_time))
        train_loss, train_acc, n_batch = 0, 0, 0
        for X_train_a, y_train_a in tl.iterate.minibatches(
                                X_train, y_train, batch_size, shuffle=True):
            feed_dict = {x: X_train_a, y_: y_train_a}
            err, ac = sess.run([cost, acc], feed_dict=feed_dict)
            train_loss += err; train_acc += ac; n_batch += 1
        print("   train loss: %f" % (train_loss/ n_batch))
        print("   train acc: %f" % (train_acc/ n_batch))
        test_loss, test_acc, n_batch = 0, 0, 0
        for X_test_a, y_test_a in tl.iterate.minibatches(
                                    X_test, y_test, batch_size, shuffle=True):
            feed_dict = {x: X_test_a, y_: y_test_a}
            err, ac = sess.run([cost, acc], feed_dict=feed_dict)
            test_loss += err; test_acc += ac; n_batch += 1
        print("   test loss: %f" % (test_loss/ n_batch))
        print("   test acc: %f" % (test_acc/ n_batch))

            
#sess.close()

if __name__ =='__main__':
    print(y_test[0])
    
