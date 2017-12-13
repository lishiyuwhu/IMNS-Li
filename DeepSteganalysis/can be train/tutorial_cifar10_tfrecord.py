#! /usr/bin/python
# -*- coding: utf8 -*-


import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import *
import numpy as np
import time
from PIL import Image
import os
import io



def read_and_decode(filename, is_train=None):
    """ Return tensor to read from TFRecord """
    filename_queue = tf.train.string_input_producer([filename])
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'label': tf.FixedLenFeature([], tf.int64),
                                           'img_raw' : tf.FixedLenFeature([], tf.string),
                                       })
    img = tf.decode_raw(features['img_raw'], tf.uint8)
    img = tf.reshape(img, [32, 32, 3])

    label = tf.cast(features['label'], tf.int32)
    img = tf.cast(img, tf.float32)
    return img, label

batch_size = 128

with tf.device('/cpu:0'):
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    # prepare data in cpu
    x_train_, y_train_ = read_and_decode("boossbase.train")
    x_train_batch, y_train_batch = tf.train.shuffle_batch([x_train_, y_train_],
        batch_size=batch_size, capacity=2000, min_after_dequeue=1000)#, num_threads=32) # set the number of threads here

    x_test_, y_test_ = read_and_decode("boossbase.test")
    x_test_batch, y_test_batch = tf.train.batch([x_test_, y_test_],
        batch_size=batch_size, capacity=50000)#, num_threads=32)

    def model(x_crop, y_, reuse):
        """ For more simplified CNN APIs, check tensorlayer.org """
        W_init = tf.truncated_normal_initializer(stddev=5e-2)
        W_init2 = tf.truncated_normal_initializer(stddev=0.04)
        b_init2 = tf.constant_initializer(value=0.1)
        with tf.variable_scope("model", reuse=reuse):
            tl.layers.set_name_reuse(reuse)
            net = InputLayer(x_crop, name='input')
            net = Conv2d(net, 64, (5, 5), (1, 1), act=tf.nn.relu,
                        padding='SAME', W_init=W_init, name='cnn1')
            net = MaxPool2d(net, (3, 3), (2, 2), padding='SAME',name='pool1')
            net = Conv2d(net, 64, (5, 5), (1, 1), act=tf.nn.relu,
                        padding='SAME', W_init=W_init, name='cnn2')
            net = MaxPool2d(net, (3, 3), (2, 2), padding='SAME',name='pool2')
            net = FlattenLayer(net, name='flatten')                             # output: (batch_size, 2304)
            net = DenseLayer(net, n_units=384, act=tf.nn.relu,
                        W_init=W_init2, b_init=b_init2, name='d1relu')           # output: (batch_size, 384)
            net = DenseLayer(net, n_units=192, act=tf.nn.relu,
                        W_init=W_init2, b_init=b_init2, name='d2relu')           # output: (batch_size, 192)
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

            # correct_prediction = tf.equal(tf.argmax(tf.nn.softmax(y), 1), y_)
            correct_prediction = tf.equal(tf.cast(tf.argmax(y, 1), tf.int32), y_)
            acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

            return net, cost, acc

    with tf.device('/gpu:0'): # <-- remove it if you don't have GPU
        ## using local response normalization
        network, cost, acc, = model(x_train_batch, y_train_batch, False)
        _, cost_test, acc_test = model(x_test_batch, y_test_batch, True)

    ## train
    n_epoch = 50000
    learning_rate = 0.0001
    print_freq = 1
    # n_step_epoch = int(len(y_train)/batch_size)
    n_step_epoch = int(64000 / batch_size)
    n_step = n_epoch * n_step_epoch

    with tf.device('/gpu:0'):   # <-- remove it if you don't have GPU
        train_op = tf.train.AdamOptimizer(learning_rate, beta1=0.9, beta2=0.999,
            epsilon=1e-08, use_locking=False).minimize(cost)

    tl.layers.initialize_global_variables(sess)

    network.print_params(False)
    network.print_layers()

    print('   learning_rate: %f' % learning_rate)
    print('   batch_size: %d' % batch_size)
    print('   n_epoch: %d, step in an epoch: %d, total n_step: %d' % (n_epoch, n_step_epoch, n_step))

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    step = 0
    for epoch in range(n_epoch):
        start_time = time.time()
        train_loss, train_acc, n_batch = 0, 0, 0
        for s in range(n_step_epoch):

            err, ac, _ = sess.run([cost, acc, train_op])
            step += 1; train_loss += err; train_acc += ac; n_batch += 1

        if epoch + 1 == 1 or (epoch + 1) % print_freq == 0:
            print("Epoch %d : Step %d-%d of %d took %fs" % (epoch, step, step + n_step_epoch, n_step, time.time() - start_time))
            print("   train loss: %f" % (train_loss/ n_batch))
            print("   train acc: %f" % (train_acc/ n_batch))

            test_loss, test_acc, n_batch = 0, 0, 0
            # for _ in range(int(len(y_test)/batch_size)):
            for _ in range(int(16000 / batch_size)):
                err, ac = sess.run([cost_test, acc_test])
                test_loss += err; test_acc += ac; n_batch += 1
            print("   test loss: %f" % (test_loss/ n_batch))
            print("   test acc: %f" % (test_acc/ n_batch))

    coord.request_stop()
    coord.join(threads)
    sess.close()
