 net = InputLayer(x_crop, name='inputlayer')
            net = Conv2d(net, 1, (5, 5), (1, 1), act=tf.identity,
                         padding='VALID', W_init=high_pass_filter, name='HighPass')
            net = Conv2d(net, 64, (5, 5), (2, 2), act=tf.nn.relu,
                         padding='VALID', W_init=W_init, name='trainCONV1')
            # net = tl.layers.MaxPool2d(net, (2, 2), (2, 2), padding='SAME', name='pool1')
            net = Conv2d(net, 16, (5, 5), (2, 2), act=tf.nn.relu,
                         padding='VALID', W_init=W_init, name='trainCONV2')
            # net = tl.layers.MaxPool2d(net, (2, 2), (2, 2), padding='SAME', name='pool2')
            net = FlattenLayer(net, name='trainFlatten')
            net = DenseLayer(net, n_units=500, act=tf.nn.relu,
                             W_init=W_init2, b_init=b_init2, name='trainFC1')
            net = DenseLayer(net, n_units=500, act=tf.nn.relu,
                             W_init=W_init2, b_init=b_init2, name='trainFC2')
            net = DenseLayer(net, n_units=2, act=tf.identity,
                             W_init=W_init, name='trainOutput')