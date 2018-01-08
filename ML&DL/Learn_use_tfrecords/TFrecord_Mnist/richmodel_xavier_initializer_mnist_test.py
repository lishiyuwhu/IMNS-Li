#! /usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2017/01/08 15:41
# @Author  : Shiyu Li
# @Software: PyCharm


import tensorlayer as tl
from tensorlayer.layers import *
import numpy as np
import time
import richmodel_filter as rm


def filter_33to55(array33):
    '''
    pad 3×3 to be the 5×5 with zeros

    example:

    In [8]:a
    Out[8]:
    array([[-1.,  2., -1.],
       [ 2., -4.,  2.],
       [-1.,  2., -1.]], dtype=float32)

    In [9]: filter_33to55(a)
    Out[9]:
    array([[ 0.,  0.,  0.,  0.,  0.],
       [ 0., -1.,  2., -1.,  0.],
       [ 0.,  2., -4.,  2.,  0.],
       [ 0., -1.,  2., -1.,  0.],
       [ 0.,  0.,  0.,  0.,  0.]], dtype=float32)

    '''

    a = np.zeros(3, dtype=np.float32)
    b = np.zeros(5, dtype=np.float32)

    array33 = np.column_stack((a, array33))
    array33 = np.column_stack((array33, a))
    array33 = np.row_stack((b, array33))
    array33 = np.row_stack((array33, b))

    return array33


def read_and_decode(filename, img_shape):
    filename_queue = tf.train.string_input_producer([filename])
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'label': tf.FixedLenFeature([], tf.int64),
                                           'img_raw': tf.FixedLenFeature([], tf.string),
                                       })
    img = tf.decode_raw(features['img_raw'], tf.uint8)
    img = tf.reshape(img, img_shape)
    img = tf.cast(img, tf.float32)  # if you want to use tfrecords as input.
    label = tf.cast(features['label'], tf.int32)
    return img, label


train_file = "train_Mnist_pgm.tfrecords"
test_file = "test_Mnist_pgm.tfrecords"
img_shape = [28, 28, 1]

## train params

train_num = sum(1 for _ in tf.python_io.tf_record_iterator(train_file))
test_num = sum(1 for _ in tf.python_io.tf_record_iterator(test_file))

batch_size = 128
n_epoch = 500
learning_rate = 0.01
print_freq = 1
n_step_epoch = int(train_num / batch_size)
n_step = n_epoch * n_step_epoch

print('   learning_rate: %f' % learning_rate)
print('   batch_size: %d' % batch_size)
print('   n_epoch: %d, step in an epoch: %d, total n_step: %d' % (n_epoch, n_step_epoch, n_step))

with tf.device('/cpu:0'):
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    # prepare data in cpu

    x_train_, y_train_ = read_and_decode(train_file, img_shape)
    x_train_batch, y_train_batch = tf.train.shuffle_batch([x_train_, y_train_],
                                                          batch_size=batch_size, capacity=2000,
                                                          min_after_dequeue=1000)  # , num_threads=32) # set the number of threads here

    x_test_, y_test_ = read_and_decode(test_file, img_shape)
    x_test_batch, y_test_batch = tf.train.batch([x_test_, y_test_],
                                                batch_size=batch_size, capacity=50000)  # , num_threads=32)


    def model(x_crop, y_, reuse):

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
            net = PoolLayer(net,
                            ksize=[1, 2, 2, 1],
                            strides=[1, 2, 2, 1],
                            padding='VALID',
                            pool=tf.nn.avg_pool,
                            name='layer4_pool')
            net = Conv2dLayer(net,
                              act=tf.nn.relu,
                              shape=[5, 5, 30, 32],
                              strides=[1, 1, 1, 1],
                              padding='VALID',
                              W_init=tf.contrib.layers.xavier_initializer(),
                              b_init=tf.constant_initializer(value=0.2),
                              name='layer5_conv')
            net = PoolLayer(net,
                            ksize=[1, 3, 3, 1],
                            strides=[1, 2, 2, 1],
                            padding='VALID',
                            pool=tf.nn.avg_pool,
                            name='layer5_pool')
            net = FlattenLayer(net, name='Flatten')
            net = DenseLayer(net,
                             n_units=100,
                             act=tf.nn.relu,
                             name='Fully_connected')
            net = DenseLayer(net,
                             n_units=10,
                             act=tf.identity,
                             name='Net_output')
        y = net.outputs
        cost = tl.cost.cross_entropy(y, y_, name='cost')
        correct_prediction = tf.equal(tf.cast(tf.argmax(y, 1), tf.int32), y_)
        acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        return net, cost, acc


    with tf.device('/gpu:0'):
        network, cost, acc, = model(x_train_batch, y_train_batch, False)
        _, cost_test, acc_test = model(x_test_batch, y_test_batch, True)

    train_vars = network.all_params

    with tf.device('/gpu:0'):  # <-- remove it if you don't have GPU
        train_op = tf.train.AdamOptimizer(
            learning_rate, use_locking=False).minimize(cost, var_list=train_vars)

    tl.layers.initialize_global_variables(sess)

    network.print_params(False)
    network.print_layers()

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