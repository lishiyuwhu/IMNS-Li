#! /usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2017/12/16 10:44
# @Author  : Shiyu Li
# @Software: PyCharm
# https://github.com/ritchieng/wideresnet-tensorlayer


import tensorlayer as tl
from tensorlayer.layers import *
import numpy as np
import time
import os

## Download data, and convert to TFRecord format, see ```tutorial_tfrecord.py```
X_train, y_train, X_test, y_test = tl.files.load_cifar10_dataset(
    shape=(-1, 32, 32, 3), plotable=False)

# X_train = np.asarray(X_train, dtype=np.float32)
# y_train = np.asarray(y_train, dtype=np.int64)
# X_test = np.asarray(X_test, dtype=np.float32)
# y_test = np.asarray(y_test, dtype=np.int64)

print('X_train.shape', X_train.shape)  # (50000, 32, 32, 3)
print('y_train.shape', y_train.shape)  # (50000,)
print('X_test.shape', X_test.shape)  # (10000, 32, 32, 3)
print('y_test.shape', y_test.shape)  # (10000,)
print('X %s   y %s' % (X_test.dtype, y_test.dtype))


def data_to_tfrecord(images, labels, filename):
    """ Save data into TFRecord """
    if os.path.isfile(filename):
        print("%s exists" % filename)
        return
    print("Converting data into %s ..." % filename)
    cwd = os.getcwd()
    writer = tf.python_io.TFRecordWriter(filename)
    for index, img in enumerate(images):
        img_raw = img.tobytes()
        ## Visualize a image
        # tl.visualize.frame(np.asarray(img, dtype=np.uint8), second=1, saveable=False, name='frame', fig_idx=1236)
        label = int(labels[index])
        # print(label)
        ## Convert the bytes back to image as follow:
        # image = Image.frombytes('RGB', (32, 32), img_raw)
        # image = np.fromstring(img_raw, np.float32)
        # image = image.reshape([32, 32, 3])
        # tl.visualize.frame(np.asarray(image, dtype=np.uint8), second=1, saveable=False, name='frame', fig_idx=1236)
        example = tf.train.Example(features=tf.train.Features(feature={
            "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
            'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw])),
        }))
        writer.write(example.SerializeToString())  # Serialize To String
    writer.close()


def read_and_decode(filename, is_train=None):
    """ Return tensor to read from TFRecord """
    filename_queue = tf.train.string_input_producer([filename])
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'label': tf.FixedLenFeature([], tf.int64),
                                           'img_raw': tf.FixedLenFeature([], tf.string),
                                       })
    # You can do more image distortion here for training data
    img = tf.decode_raw(features['img_raw'], tf.float32)
    img = tf.reshape(img, [32, 32, 3])
    img = tf.cast(img, tf.float32)  # * (1. / 255) - 0.5
    # if is_train == True:
    #     # 1. Randomly crop a [height, width] section of the image.
    #     img = tf.random_crop(img, [24, 24, 3])
    #     # 2. Randomly flip the image horizontally.
    #     img = tf.image.random_flip_left_right(img)
    #     # 3. Randomly change brightness.
    #     img = tf.image.random_brightness(img, max_delta=63)
    #     # 4. Randomly change contrast.
    #     img = tf.image.random_contrast(img, lower=0.2, upper=1.8)
    #     # 5. Subtract off the mean and divide by the variance of the pixels.
    #     try: # TF 0.12+
    #         img = tf.image.per_image_standardization(img)
    #     except: # earlier TF versions
    #         img = tf.image.per_image_whitening(img)
    #
    # elif is_train == False:
    #     # 1. Crop the central [height, width] of the image.
    #     img = tf.image.resize_image_with_crop_or_pad(img, 24, 24)
    #     # 2. Subtract off the mean and divide by the variance of the pixels.
    #     try: # TF 0.12+
    #         img = tf.image.per_image_standardization(img)
    #     except: # earlier TF versions
    #         img = tf.image.per_image_whitening(img)
    # elif is_train == None:
    #     img = img

    label = tf.cast(features['label'], tf.int32)
    return img, label


# ## Save data into TFRecord files
# data_to_tfrecord(images=X_train, labels=y_train, filename="train.cifar10")
# data_to_tfrecord(images=X_test, labels=y_test, filename="test.cifar10")

batch_size = 64
# For generator
num_examples = X_train.shape[0]
index_in_epoch = 0
epochs_completed = 0

# For wide resnets
blocks_per_group = 4
widening_factor = 4

# Basic info
batch_num = 64
img_row = 32
img_col = 32
img_channels = 3
nb_classes = 10

with tf.device('/cpu:0'):
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    # prepare data in cpu
    x_train_, y_train_ = read_and_decode("train.cifar10", True)
    x_test_, y_test_ = read_and_decode("test.cifar10", False)

    x_train_batch, y_train_batch = tf.train.shuffle_batch([x_train_, y_train_],
                                                          batch_size=batch_size, capacity=2000, min_after_dequeue=1000,
                                                          num_threads=32)  # set the number of threads here
    # for testing, uses batch instead of shuffle_batch
    x_test_batch, y_test_batch = tf.train.batch([x_test_, y_test_],
                                                batch_size=batch_size, capacity=50000, num_threads=32)


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


    def model_batch_norm(x_crop, y_, reuse):
        """ Batch normalization should be placed before rectifier. """
        with tf.variable_scope("model", reuse=reuse):
            tl.layers.set_name_reuse(reuse)
            net = InputLayer(x_crop, name='input')
            net = Conv2dLayer(net,
                              act=tf.nn.relu,
                              shape=[3, 3, 3, 16],
                              strides=[1, 1, 1, 1],
                              padding='SAME',
                              name='cnn_layer_first')
            for i in range(0, blocks_per_group):
                nb_filters = 16 * widening_factor
                count = i
                net = residual_block(net, count, nb_filters=nb_filters, subsample_factor=1)

            for i in range(0, blocks_per_group):
                nb_filters = 32 * widening_factor
                if i == 0:
                    subsample_factor = 2
                else:
                    subsample_factor = 1
                count = i + blocks_per_group
                net = residual_block(net, count, nb_filters=nb_filters, subsample_factor=subsample_factor)

            for i in range(0, blocks_per_group):
                nb_filters = 64 * widening_factor
                if i == 0:
                    subsample_factor = 2
                else:
                    subsample_factor = 1
                count = i + 2 * blocks_per_group
                net = residual_block(net, count, nb_filters=nb_filters, subsample_factor=subsample_factor)

            net = tl.layers.BatchNormLayer(net,
                                           decay=0.999,
                                           epsilon=1e-05,
                                           is_train=True,
                                           name='norm_last')

            net = tl.layers.PoolLayer(net,
                                      ksize=[1, 8, 8, 1],
                                      strides=[1, 8, 8, 1],
                                      padding='VALID',
                                      pool=tf.nn.avg_pool,
                                      name='pool_last')

            net = tl.layers.FlattenLayer(net, name='flatten')

            net = tl.layers.DenseLayer(net,
                                       n_units=nb_classes,
                                       act=tf.identity,
                                       name='fc')

            y = net.outputs

            ce = tl.cost.cross_entropy(y, y_, name='cost')
            # L2 for the MLP, without this, the accuracy will be reduced by 15%.
            # L2 = 0
            # for p in tl.layers.get_variables_with_name('relu/W', True, True):
            #     L2 += tf.contrib.layers.l2_regularizer(0.004)(p)
            cost = ce #+ L2

            correct_prediction = tf.equal(tf.cast(tf.argmax(y, 1), tf.int32), y_)
            acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

            return net, cost, acc


    with tf.device('/gpu:0'):
        network, cost, acc, = model_batch_norm(x_train_batch, y_train_batch, None)
        _, cost_test, acc_test = model_batch_norm(x_test_batch, y_test_batch, True)

    ## train
    n_epoch = 5000
    learning_rate = 0.01
    print_freq = 1
    n_step_epoch = int(len(y_train) / batch_size)
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
            ## You can also use placeholder to feed_dict in data after using
            # val, l = sess.run([x_train_batch, y_train_batch])
            # tl.visualize.images2d(val, second=3, saveable=False, name='batch', dtype=np.uint8, fig_idx=2020121)
            # err, ac, _ = sess.run([cost, acc, train_op], feed_dict={x_crop: val, y_: l})
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
            for _ in range(int(len(y_test) / batch_size)):
                err, ac = sess.run([cost_test, acc_test])
                test_loss += err
                test_acc += ac
                n_batch += 1
            print("   test loss: %f" % (test_loss / n_batch))
            print("   test acc: %f" % (test_acc / n_batch))

    coord.request_stop()
    coord.join(threads)
    sess.close()
