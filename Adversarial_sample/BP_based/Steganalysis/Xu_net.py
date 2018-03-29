'''Xu, G., Wu, H.-Z., Shi, Y.Q.: Ensemble of CNNs for Steganalysis: an empirical study. 
In: Proceedings of the 4th ACM Workshop on Information Hiding and Multimedia Security. ACM (2016)

'''


import tensorlayer as tl
import tensorflow as tf
from tensorlayer.layers import *
import numpy as np
import time

def read_and_decode(filename, img_shape):
    """ Return tensor to read from TFRecord """
    filename_queue = tf.train.string_input_producer([filename])
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'label': tf.FixedLenFeature([], tf.int64),
                                           'img_raw': tf.FixedLenFeature([], tf.string),
                                       })
    img = tf.decode_raw(features['img_raw'], tf.float32)/255
    img = tf.reshape(img, img_shape)
    label = tf.cast(features['label'], tf.int32)
    return img, label



model_file_name = "Xu_net.npz"
resume = False # load model, resume from previous checkpoint?





batch_size = 128
train_file = "train_own_BossBase-256x256.tfrecords"
test_file =  "test_own_BossBase-256x256.tfrecords"
img_shape= [256,256,1]






with tf.device('/cpu:0'):
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    # prepare data in cpu

    x_train_, y_train_ = read_and_decode(train_file, img_shape)
    x_train_batch, y_train_batch = tf.train.shuffle_batch([x_train_, y_train_],
                                                          batch_size=batch_size, capacity=2000,min_after_dequeue=1000)  # , num_threads=32) # set the number of threads here

    x_test_, y_test_ = read_and_decode(test_file, img_shape)
    x_test_batch, y_test_batch = tf.train.batch([x_test_, y_test_],
                                                batch_size=batch_size, capacity=50000)  # , num_threads=32)



    def model(x_crop, y_, reuse):
        F0 = np.array([[-1, 2, -2, 2, -1],
                       [2, -6, 8, -6, 2],
                       [-2, 8, -12, 8, -2],
                       [2, -6, 8, -6, 2],
                       [-1, 2, -2, 2, -1]], dtype=np.float32)
        F0 = F0 / 12.
        # assign numpy array to constant_initalizer and pass to get_variable
        high_pass_filter = tf.constant_initializer(value=F0, dtype=tf.float32)

        W_init = tf.truncated_normal_initializer(stddev=1 / 192.0)
        W_init2 = tf.truncated_normal_initializer(stddev=0.04)
        b_init2 = tf.constant_initializer(value=0.1)
        with tf.variable_scope("model", reuse=reuse):
            tl.layers.set_name_reuse(reuse)
            net = InputLayer(x_crop, name='inputlayer')
            net = Conv2d(net, 1, (5, 5), (1, 1), act=tf.identity,
                         padding='SAME', W_init=high_pass_filter, name='KV')

            net = Conv2dLayer(net,
                              act=tf.identity,
                              shape=[5, 5, 1, 8],
                              strides=[1, 1, 1, 1],
                              padding='SAME',
                              W_init=W_init,
                              b_init=b_init,
                              name='layer1_conv')
            net = LambdaLayer(net, lambda x: tf.abs(x), name='abs_layer')
            net = BatchNormLayer(net,
                                 act=tf.nn.tanh,
                                 is_train=is_train,
                                 name='bn1')
            net = PoolLayer(net,
                            ksize=[1, 5, 5, 1],
                            strides=[1, 2, 2, 1],
                            padding='SAME',
                            pool=tf.nn.avg_pool,
                            name='pool1')
            # ========
            net = Conv2dLayer(net,
                              act=tf.identity,
                              shape=[5, 5, 8, 16],
                              strides=[1, 1, 1, 1],
                              padding='SAME',
                              W_init=W_init,
                              b_init=b_init,
                              name='layer2_conv')
            net = BatchNormLayer(net,
                                 act=tf.nn.tanh,
                                 is_train=is_train,
                                 name='bn2')
            net = PoolLayer(net,
                            ksize=[1, 5, 5, 1],
                            strides=[1, 2, 2, 1],
                            padding='SAME',
                            pool=tf.nn.avg_pool,
                            name='pool2')
            # ========
            net = Conv2dLayer(net,
                              act=tf.identity,
                              shape=[1, 1, 16, 32],
                              strides=[1, 1, 1, 1],
                              padding='SAME',
                              W_init=W_init,
                              b_init=b_init,
                              name='layer3_conv')
            net = BatchNormLayer(net,
                                 act=tf.nn.relu,
                                 is_train=is_train,
                                 name='bn3')
            net = PoolLayer(net,
                            ksize=[1, 5, 5, 1],
                            strides=[1, 2, 2, 1],
                            padding='SAME',
                            pool=tf.nn.avg_pool,
                            name='pool3')
            # ========
            net = Conv2dLayer(net,
                              act=tf.identity,
                              shape=[1, 1, 32, 64],
                              strides=[1, 1, 1, 1],
                              padding='SAME',
                              W_init=W_init,
                              b_init=b_init,
                              name='layer4_conv')
            net = BatchNormLayer(net,
                                 act=tf.nn.relu,
                                 is_train=is_train,
                                 name='bn4')
            net = PoolLayer(net,
                            ksize=[1, 7, 7, 1],
                            strides=[1, 2, 2, 1],
                            padding='SAME',
                            pool=tf.nn.avg_pool,
                            name='pool4')
            # ========
            net = Conv2dLayer(net,
                              act=tf.identity,
                              shape=[1, 1, 64, 128],
                              strides=[1, 1, 1, 1],
                              padding='SAME',
                              W_init=W_init,
                              b_init=b_init,
                              name='layer5_conv')
            net = BatchNormLayer(net,
                                 act=tf.nn.relu,
                                 is_train=is_train,
                                 name='bn5')
            net = PoolLayer(net,
                            ksize=[1, 7, 7, 1],
                            strides=[1, 2, 2, 1],
                            padding='SAME',
                            pool=tf.nn.avg_pool,
                            name='pool5')
            # ========
            net = Conv2dLayer(net,
                              act=tf.identity,
                              shape=[1, 1, 128, 256],
                              strides=[1, 1, 1, 1],
                              padding='SAME',
                              W_init=W_init,
                              b_init=b_init,
                              name='layer6_conv')
            net = BatchNormLayer(net,
                                 act=tf.nn.relu,
                                 is_train=is_train,
                                 name='bn6')
            net = PoolLayer(net,
                            ksize=[1, 8, 8, 1],
                            strides=[1, 1, 1, 1],
                            padding='SAME',
                            pool=tf.nn.avg_pool,
                            name='pool6')

            net = FlattenLayer(net, name='trainFlatten')
            net = DenseLayer(net, n_units=512, act=tf.nn.relu,
                             W_init=W_init2, b_init=b_init, name='trainFC2')
            net = DenseLayer(net, n_units=2, act=tf.identity,
                             W_init=W_init2, name='trainOutput')
        y = net.outputs
        cost = tl.cost.cross_entropy(y, y_, name='cost')

        correct_prediction = tf.equal(tf.cast(tf.argmax(y, 1), tf.int32), y_)
        acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        return net, cost, acc



    with tf.device('/gpu:0'):
        network, cost, acc, = model(x_train_batch, y_train_batch, False)
        _, cost_test, acc_test = model(x_test_batch, y_test_batch, True)

    ## train
    train_num = sum(1 for _ in tf.python_io.tf_record_iterator(train_file))
    test_num = sum(1 for _ in tf.python_io.tf_record_iterator(test_file))

    n_epoch = 500
    learning_rate = 0.01
    print_freq = 1
    n_step_epoch = int(train_num / batch_size)
    n_step = n_epoch * n_step_epoch

    train_vars = network.all_params[2:]

    with tf.device('/gpu:0'):  # <-- remove it if you don't have GPU
        train_op = tf.train.AdamOptimizer(
            learning_rate, use_locking=False).minimize(cost, var_list=train_vars)

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

        if (epoch+1)%50 == 0 :
            print("Save model " + "!" * 10)
            model_name = model_file_name + '_' + str(epoch)
            tl.files.save_npz(network.all_params , name=model_name)

    coord.request_stop()
    coord.join(threads)
    sess.close()
