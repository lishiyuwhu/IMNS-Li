#! /usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2017/11/23 14:42
# @Author  : Shiyu Li
# @Software: PyCharm Community Edition

import tensorflow as tf
import tensorlayer as tl
import numpy as np
import time

NUM_DIGITS = 10

def bi_encode(i, num_digits):
    return np.array([i >> d & 1 for d in range(num_digits)][::-1])

def Fizz_Buzz_encode(i):
    if   i % 15 == 0: return 3
    elif i % 5  == 0: return 2
    elif i % 3  == 0: return 1
    else:             return 0

# def Fizz_Buzz_encode(i):
#     if   i % 15 == 0: return np.array([0, 0, 0, 1])
#     elif i % 5  == 0: return np.array([0, 0, 1, 0])
#     elif i % 3  == 0: return np.array([0, 1, 0, 0])
#     else:             return np.array([1, 0, 0, 0])
#

def MLP_model():
    X_train = np.array([bi_encode(i, NUM_DIGITS) for i in range(101, 2 ** NUM_DIGITS)])
    y_train = np.array([Fizz_Buzz_encode(i) for i in range(101, 2 ** NUM_DIGITS)])

    X_test = np.array([bi_encode(i, NUM_DIGITS) for i in range(101)])
    y_test = np.array([Fizz_Buzz_encode(i) for i in range(101)])

    print('X_train.shape', X_train.shape)
    print('y_train.shape', y_train.shape)
    print('X_test.shape', X_test.shape)
    print('y_test.shape', y_test.shape)
    print('X %s   y %s' % (X_test.dtype, y_test.dtype))

    sess = tf.InteractiveSession()

    # placeholder
    x = tf.placeholder(tf.float32, shape=[None, NUM_DIGITS], name='x')
    y_ = tf.placeholder(tf.int64, shape=[None, ], name='y_')

    net = tl.layers.InputLayer(x, name='input_layer')
    #net = tl.layers.DenseLayer(net, n_units=100, act=tf.nn.relu, name='relu1')
    net = tl.layers.DenseLayer(net, n_units=1000, act=tf.nn.relu, name='relu2')
    net = tl.layers.DenseLayer(net, n_units=4, act=tf.identity, name='output_layer')
    y = net.outputs

    cost = tl.cost.cross_entropy(y, y_, name='xentropy')

    # tl.cost.cross_entropy is tf.nn.sparse_softmax_cross_entropy_with_logits
    #cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(y, y_,name='xentropy'))

    correct_prediction = tf.equal(tf.argmax(y, 1), y_)
    acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # train
    n_epoch = 200
    batch_size = 128
    learning_rate = 0.001
    print_freq = 5
    train_op = tf.train.AdamOptimizer(learning_rate, beta1=0.9, beta2=0.999,
                                epsilon=1e-08, use_locking=False).minimize(cost)

    tl.layers.initialize_global_variables(sess)

    net.print_params()
    net.print_layers()

    print('   learning_rate: %f' % learning_rate)
    print('   batch_size: %d' % batch_size)

    for epoch in range(n_epoch):
        start_time = time.time()
        for X_train_a, y_train_a in tl.iterate.minibatches(X_train, y_train, batch_size, shuffle=True):
            feed_dict = {x: X_train_a, y_: y_train_a}
            sess.run(train_op, feed_dict=feed_dict)

        if epoch + 1 == 1 or (epoch+1)%print_freq==0:
            print("Epoch %d of %d took %fs" % (epoch + 1, n_epoch, time.time() - start_time))

            train_loss, train_acc, n_batch = 0, 0, 0
            for X_train_a, y_train_a in tl.iterate.minibatches(
                                    X_train, y_train, batch_size, shuffle=True):
                err, ac = sess.run([cost, acc], feed_dict={x: X_train_a, y_: y_train_a})
                train_loss += err
                train_acc += ac
                n_batch += 1
            print("   train loss: %f" % (train_loss/ n_batch))
            print("   train acc: %f" % (train_acc/ n_batch))



            test_loss, test_acc, n_batch = 0, 0, 1
            err, ac = sess.run([cost, acc], feed_dict={x: X_test, y_: y_test})
            test_loss += err
            test_acc += ac
            #n_batch += 1
            print("   test loss: %f" % (test_loss / n_batch))
            print("   test acc: %f" % (test_acc / n_batch))

    sess.close()

if __name__ == '__main__':
    sess = tf.InteractiveSession()
    MLP_model()