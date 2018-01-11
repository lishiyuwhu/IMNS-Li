#! /usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2017/12/17 14:36
# @Author  : Shiyu Li
# @Software: PyCharm
# Thanks to @ritchieng!


import tensorlayer as tl
from tensorlayer.layers import *
import numpy as np
import time
import os


def read_and_decode(filename):
    filename_queue = tf.train.string_input_producer([filename])
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'label': tf.FixedLenFeature([], tf.int64),
                                           'img_raw': tf.FixedLenFeature([], tf.string),
                                       })

    img = tf.decode_raw(features['img_raw'], tf.uint8)
    img = tf.reshape(img, [28, 28, 1])
    img = tf.cast(img, tf.float32)
    label = tf.cast(features['label'], tf.int32)
    return img, label



trainfile = 'train_Mnist_pgm.tfrecords'
testfile = 'test_Mnist_pgm.tfrecords'




num_examples = sum(1 for _ in tf.python_io.tf_record_iterator(trainfile))
test_num = sum(1 for _ in tf.python_io.tf_record_iterator(testfile))
## For convenience.
# num_examples = 64000
train_num = num_examples
# test_num = 16000
batch_size = 128

with tf.device('/cpu:0'):
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    # prepare data in cpu
    x_train_, y_train_ = read_and_decode(trainfile)
    x_test_, y_test_ = read_and_decode(testfile)

    x_train_batch, y_train_batch = tf.train.shuffle_batch([x_train_, y_train_],
                                                          batch_size=batch_size, capacity=2000, min_after_dequeue=1000,
                                                          num_threads=2)
    x_test_batch, y_test_batch = tf.train.batch([x_test_, y_test_],
                                                batch_size=batch_size, capacity=50000, num_threads=2)

    def model(x_crop, y_, reuse):
        W_init = tf.truncated_normal_initializer(stddev=1 / 192.0)
        W_init2 = tf.truncated_normal_initializer(stddev=0.04)
        b_init2 = tf.constant_initializer(value=0.1)
        F55 = rm.list_33
        F33 = rm.list_33
        Filter = F55 + F33
        Filter = np.array(Filter, dtype=np.float32)

        Richmodel_num_55 = len(F55)
        Richmodel_num_33 = len(F33)

        Richmodel_num = Richmodel_num_33 + Richmodel_num_55
        rich_filter = tf.constant_initializer(value=Filter, dtype=tf.float32)

        with tf.variable_scope("model", reuse=reuse):
            tl.layers.set_name_reuse(reuse)
            net = InputLayer(x_crop, name='inputlayer')
            net = Conv2dLayer(net,
                              act=tf.identity,
                              shape=[5, 5, 1, Richmodel_num],
                              strides=[1, 1, 1, 1],
                              padding='VALID',
                              W_init=rich_filter,
                              b_init=tf.constant_initializer(value=0.0),
                              name='layer1_richmodel_filter')
            net = Conv2dLayer(net,
                              act=tf.nn.relu,
                              shape=[3, 3, Richmodel_num, 30],
                              strides=[1, 1, 1, 1],
                              padding='VALID',
                              W_init=tf.contrib.layers.xavier_initializer(),
                              b_init=tf.constant_initializer(value=0.2),
                              name='layer2_conv')
            net = Conv2d(net, 64, (5, 5), (2, 2), act=tf.nn.relu,
                         padding='VALID', W_init=W_init, name='train_CONV1')
            net = FlattenLayer(net, name='train_Flatten')
            net = DenseLayer(net, n_units=100, act=tf.nn.relu,
                             W_init=W_init2, b_init=b_init2, name='train_FC1')
            net = DenseLayer(net, n_units=10, act=tf.identity,
                             W_init=W_init, name='train_Output')
        y = net.outputs
        ce = tl.cost.cross_entropy(y, y_, name='cost')
        L2 = 0
        for p in tl.layers.get_variables_with_name('FC', True, True):
            L2 += tf.contrib.layers.l2_regularizer(0.004)(p)
        cost = ce + L2

        correct_prediction = tf.equal(tf.cast(tf.argmax(y, 1), tf.int32), y_)
        acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        return net, cost, acc


    with tf.device('/gpu:0'):
        network, cost, acc, = model(x_train_batch, y_train_batch, None)
        _, cost_test, acc_test = model(x_test_batch, y_test_batch, True)

    ## train
    n_epoch = 5000
    learning_rate = 0.01
    print_freq = 1
    n_step_epoch = int(train_num / batch_size)
    n_step = n_epoch * n_step_epoch

    with tf.device('/gpu:0'):
        # train_op = tf.train.AdamOptimizer(learning_rate, beta1=0.9, beta2=0.999,
        #     epsilon=1e-08, use_locking=False).minimize(cost)
        train_op = tf.train.GradientDescentOptimizer(
            learning_rate, use_locking=False).minimize(cost, var_list=network.all_params)

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
            step += 1
            train_loss += err
            train_acc += ac
            n_batch += 1

        if epoch + 1 == 1 or (epoch + 1) % print_freq == 0:
            print("Epoch %d : Step %d-%d of %d took %fs" % (
                epoch, step, step + n_step_epoch, n_step, time.time() - start_time))
            print("   train loss: %f" % (train_loss / n_batch))
            print("   train acc: %f" % (train_acc / n_batch))

            test_loss, test_acc, n_batch = 0, 0, 0
            for _ in range(int(test_num / batch_size)):
                err, ac = sess.run([cost_test, acc_test])
                test_loss += err
                test_acc += ac
                n_batch += 1
            print("   test loss: %f" % (test_loss / n_batch))
            print("   test acc: %f" % (test_acc / n_batch))

    coord.request_stop()
    coord.join(threads)
    sess.close()
