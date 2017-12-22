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


#
# def read_and_decode(filename, is_train=None):
#     filename_queue = tf.train.string_input_producer([filename])
#     reader = tf.TFRecordReader()
#     _, serialized_example = reader.read(filename_queue)
#     features = tf.parse_single_example(serialized_example,
#                                        features={
#                                            'label': tf.FixedLenFeature([], tf.int64),
#                                            'img_raw': tf.FixedLenFeature([], tf.string),
#                                        })
#     img = tf.decode_raw(features['img_raw'], tf.uint8)
#     img = tf.reshape(img, [256, 256, 1])
#     img = tf.cast(img, tf.float32) * (1. / 255) - 0.5
#     if is_train == True:
#         # 1. Randomly crop a [height, width] section of the image.
#         img = tf.random_crop(img, [128, 128, 1])
#         # 2. Randomly flip the image horizontally.
#         img = tf.image.random_flip_left_right(img)
#         # 3. Randomly change brightness.
#         img = tf.image.random_brightness(img, max_delta=63)
#         # 4. Randomly change contrast.
#         img = tf.image.random_contrast(img, lower=0.2, upper=1.8)
#         # 5. Subtract off the mean and divide by the variance of the pixels.
#         img = tf.image.per_image_standardization(img)
#
#
#     elif is_train == False:
#         # 1. Crop the central [height, width] of the image.
#         img = tf.image.resize_image_with_crop_or_pad(img, 128, 128)
#         # 2. Subtract off the mean and divide by the variance of the pixels.
#         img = tf.image.per_image_standardization(img)
#
#     elif is_train == None:
#         img = img
#
#     label = tf.cast(features['label'], tf.int32)
#     return img, label


trainfile = 'train_Mnist_pgm.tfrecords'
testfile = 'test_Mnist_pgm.tfrecords'




num_examples = sum(1 for _ in tf.python_io.tf_record_iterator(trainfile))
test_num = sum(1 for _ in tf.python_io.tf_record_iterator(testfile))
## For convenience.
# num_examples = 64000
train_num = num_examples
# test_num = 16000

index_in_epoch = 0
epochs_completed = 0

# For wide resnets
blocks_per_group = 4
widening_factor = 4

# Basic info
batch_size = 32
batch_num = batch_size
img_row = 28
img_col = 28
img_channels = 1
nb_classes = 10

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


    def zero_pad_channels(x, pad=0):
        """
        Function for Lambda layer
        """
        pattern = [[0, 0], [0, 0], [0, 0], [pad - pad // 2, pad // 2]]
        return tf.pad(x, pattern)


    def residual_block(x, count, nb_filters=16, subsample_factor=1):
        prev_nb_channels = x.outputs.get_shape().as_list()[3]

        if subsample_factor > 1:
            subsample = [1, subsample_factor, subsample_factor, 1]
            # shortcut: subsample + zero-pad channel dim
            name_pool = 'pool_layer' + str(count)
            shortcut = tl.layers.PoolLayer(x,
                                           ksize=subsample,
                                           strides=subsample,
                                           padding='VALID',
                                           pool=tf.nn.avg_pool,
                                           name=name_pool)

        else:
            subsample = [1, 1, 1, 1]
            # shortcut: identity
            shortcut = x

        if nb_filters > prev_nb_channels:
            name_lambda = 'lambda_layer' + str(count)
            shortcut = tl.layers.LambdaLayer(
                shortcut,
                zero_pad_channels,
                fn_args={'pad': nb_filters - prev_nb_channels},
                name=name_lambda)

        name_norm = 'norm' + str(count)
        y = tl.layers.BatchNormLayer(x,
                                     decay=0.999,
                                     epsilon=1e-05,
                                     is_train=True,
                                     name=name_norm)

        name_conv = 'conv_layer' + str(count)
        y = tl.layers.Conv2dLayer(y,
                                  act=tf.nn.relu,
                                  shape=[3, 3, prev_nb_channels, nb_filters],
                                  strides=subsample,
                                  padding='SAME',
                                  name=name_conv)

        name_norm_2 = 'norm_second' + str(count)
        y = tl.layers.BatchNormLayer(y,
                                     decay=0.999,
                                     epsilon=1e-05,
                                     is_train=True,
                                     name=name_norm_2)

        prev_input_channels = y.outputs.get_shape().as_list()[3]
        name_conv_2 = 'conv_layer_second' + str(count)
        y = tl.layers.Conv2dLayer(y,
                                  act=tf.nn.relu,
                                  shape=[3, 3, prev_input_channels, nb_filters],
                                  strides=[1, 1, 1, 1],
                                  padding='SAME',
                                  name=name_conv_2)

        name_merge = 'merge' + str(count)
        out = tl.layers.ElementwiseLayer([y, shortcut],
                                         combine_fn=tf.add,
                                         name=name_merge)
        return out


    def wide_res_model(x_crop, y_, reuse):
        W_init = tf.truncated_normal_initializer(stddev=1 / 192.0)
        W_init2 = tf.truncated_normal_initializer(stddev=0.04)
        b_init2 = tf.constant_initializer(value=0.1)
        with tf.variable_scope("model", reuse=reuse):
            tl.layers.set_name_reuse(reuse)
            net = InputLayer(x_crop, name='train_input_layer')
            net = Conv2d(net, 64, (5, 5), (2, 2), act=tf.nn.relu,
                         padding='VALID', W_init=W_init, name='train_CONV1')
            net = Conv2d(net, 16, (3, 3), (2, 2), act=tf.nn.relu,
                         padding='VALID', W_init=W_init, name='train_CONV2')
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
        network, cost, acc, = wide_res_model(x_train_batch, y_train_batch, None)
        _, cost_test, acc_test = wide_res_model(x_test_batch, y_test_batch, True)

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
