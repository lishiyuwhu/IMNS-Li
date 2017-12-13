#! /usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2017/12/12 15:03
# @Author  : Shiyu Li
# @Software: PyCharm

import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import *
import numpy as np
import time
from PIL import Image
import os
import io


cwd = os.getcwd()
root = cwd + "/train"
x_train = []
y_train = []
i=0
for className in os.listdir(root):
    label = int(className[0])
    classPath = root + "/" + className + "/"
    for parent, dirnames, filenames in os.walk(classPath):
        for filename in filenames:
            imgPath = classPath + "/" + filename
            img = Image.open(imgPath)
            data = img.getdata()
            data = np.array(data, dtype=np.float32) / 255.0
            x_train.append(data)
            y_train.append(label)

x_val = []
y_val = []
root = cwd + "/test"
for className in os.listdir(root):
    label = int(className[0])
    classPath = root + "/" + className + "/"
    for parent, dirnames, filenames in os.walk(classPath):
        for filename in filenames:
            imgPath = classPath + "/" + filename
            img = Image.open(imgPath)
            data = img.getdata()
            data = np.array(data, dtype=np.float32) / 255.0
            x_val.append(data)
            y_val.append(label)



X_train = np.array(x_train)
y_train = np.asarray(y_train, dtype=np.int32)
X_val = np.array(x_val)
y_val = np.asarray(y_val, dtype=np.int32)

print('X_train.shape', X_train.shape)
print('y_train.shape', y_train.shape)

sess = tf.InteractiveSession()

# placeholder
x = tf.placeholder(tf.float32, shape=[None, 65536], name='x')
y_ = tf.placeholder(tf.int64, shape=[None, ], name='y_')

network = tl.layers.InputLayer(x, name='input')
network = tl.layers.DenseLayer(network, n_units=800,
                               act=tf.nn.relu, name='relu1')
network = tl.layers.DenseLayer(network, n_units=800,
                               act=tf.nn.relu, name='relu2')
network = tl.layers.DenseLayer(network, n_units=10,
                               act=tf.identity,
                               name='output')

y = network.outputs
cost = tl.cost.cross_entropy(y, y_, name='xentropy')
correct_prediction = tf.equal(tf.argmax(y, 1), y_)
acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
y_op = tf.argmax(tf.nn.softmax(y), 1)

params = network.all_params
# train
n_epoch = 100
batch_size = 15
learning_rate = 0.001
print_freq = 1
train_op = tf.train.AdamOptimizer(learning_rate, beta1=0.9, beta2=0.999,
                                  epsilon=1e-08, use_locking=False).minimize(cost)

tl.layers.initialize_global_variables(sess)

network.print_params()
network.print_layers()

print('   learning_rate: %f' % learning_rate)
print('   batch_size: %d' % batch_size)

for epoch in range(n_epoch):
    start_time = time.time()
    for X_train_a, y_train_a in tl.iterate.minibatches(X_train, y_train,
                                                       batch_size, shuffle=True):
        feed_dict = {x: X_train_a, y_: y_train_a}
        sess.run(train_op, feed_dict=feed_dict)

    if epoch + 1 == 1 or (epoch + 1) % print_freq == 0:
        print("Epoch %d of %d took %fs" % (epoch + 1, n_epoch, time.time() - start_time))
        train_loss, train_acc, n_batch = 0, 0, 0
        for X_train_a, y_train_a in tl.iterate.minibatches(
                X_train, y_train, batch_size, shuffle=True):
            feed_dict = {x: X_train_a, y_: y_train_a}
            err, ac = sess.run([cost, acc], feed_dict=feed_dict)
            train_loss += err;
            train_acc += ac;
            n_batch += 1
        print("   train loss: %f" % (train_loss / n_batch))
        print("   train acc: %f" % (train_acc/ n_batch))

        val_loss, val_acc, n_batch = 0, 0, 0
        for X_val_a, y_val_a in tl.iterate.minibatches(
                X_val, y_val, batch_size, shuffle=True):
            feed_dict = {x: X_val_a, y_: y_val_a}
            err, ac = sess.run([cost, acc], feed_dict=feed_dict)
            val_loss += err
            val_acc += ac
            n_batch += 1
        print("   val loss: %f" % (val_loss / n_batch))
        print("   val acc: %f" % (val_acc / n_batch))


sess.close()

# if __name__ == '__main__':
#     main()

