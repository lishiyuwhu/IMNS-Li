# -*- coding: utf-8 -*-
"""
Created on Sat May 26 09:54:15 2018

@author: Li
@Software: Spyder
"""
import os
import tensorflow as tf
import tensorlayer as tl
from attacks import deepfool
import numpy as np

model_file_name = "mnist_model.npz"

img_size = 28
img_chan = 1
n_classes = 10

resume = True



def model(x, logits=False, training=False):
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
    
    logits_ = tf.layers.dense(network, units=10, name='logits')
    y = tf.nn.softmax(logits_, name='ybar')

    if logits:
        return y, logits_
    return y
    
class Dummy:
    pass


env = Dummy()


with tf.variable_scope('model'):
    env.x = tf.placeholder(tf.float32, (None, img_size, img_size, img_chan),
                           name='x')

    env.y_ = tf.placeholder(tf.float32, (None,), name='y')
    env.training = tf.placeholder_with_default(False, (), name='mode')

    net, env.ybar, logits = model(env.x, logits=True, training=env.training)

    with tf.variable_scope('acc'):
        correct_prediction = tf.equal(tf.argmax(logits, 1), env.y_)
        env.acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    with tf.variable_scope('cost'):
        env.cost = tl.cost.cross_entropy(logits, env.y_, 'cost')

    with tf.variable_scope('train_op'):
        optimizer = tf.train.AdamOptimizer()
        env.train_op = optimizer.minimize(env.cost)


with tf.variable_scope('model', reuse=True):
    env.adv_epochs = tf.placeholder(tf.int32, (), name='adv_epochs')
    env.xadv = deepfool(model, env.x, epochs=env.adv_epochs)
    
print('\nInitializing graph')

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

if resume:
    print("Load existing model " + "!" * 10)
    tl.files.load_and_assign_npz(sess=sess, name=model_file_name, network=net)



