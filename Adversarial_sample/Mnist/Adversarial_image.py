#! /usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2018/2/28 9:30
# @Author  : Shiyu Li
# @Software: PyCharm

#! /usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2018/2/27 15:15
# @Author  : Shiyu Li
# @Software: PyCharm

import numpy as np
import tensorlayer as tl
import tensorflow as tf
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
#import time


def Review_img():
    image_list = X_train[:9]
    image_list_labels = y_train[:9]
    fig = plt.figure(1, (5., 5.))
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                     nrows_ncols=(3, 3),  # creates 2x2 grid of axes
                     axes_pad=0.3,  # pad between axes in inch.
                     )

    for i in range(len(image_list)):
        image = image_list[i].reshape(28, 28)
        grid[i].imshow(image)
        grid[i].set_title('Label: {0}'.format(image_list_labels[i].argmax()))

    plt.show()

def plot_predictions(image_list, output_probs=False, adversarial=False):
    '''
    Evaluate images against trained model and plot images.
    If adversarial == True, replace middle image title appropriately
    Return probability list if output_probs == True
    '''
    prob = y_prob.eval(feed_dict={x: image_list})
    
    
    pred_list = np.zeros(len(image_list)).astype(int)
    pct_list = np.zeros(len(image_list)).astype(int)
    
    # Setup image grid
    import math
    cols = 3
    rows = math.ceil(image_list.shape[0]/cols)
    fig = plt.figure(1, (12., 12.))
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                     nrows_ncols=(rows, cols),  # creates grid of axes
                     axes_pad=0.5,  # pad between axes in inch.
                     )
    
    # Get probs, images and populate grid
    for i in range(len(prob)):
        pred_list[i] = np.argmax(prob[i]) # for mnist index == classification
        pct_list[i] = prob.max(axis=1)[i]* 100
        
        image = image_list[i].reshape(28,28)
#        image = image_list
        
        grid[i].imshow(image)
        
        grid[i].set_title('Label: {0} \nCertainty: {1}%' \
                          .format(pred_list[i], 
                                  pct_list[i]))
        
        # Only use when plotting original, partial deriv and adversarial images
        if (adversarial) & (i % 3 == 1): 
            grid[i].set_title("Adversarial \nPartial Derivatives")
    plt.show()
    
    return prob if output_probs else None


# Mostly inspired by:
# https://codewords.recurse.com/issues/five/why-do-neural-networks-think-a-panda-is-a-vulture
def create_plot_adversarial_images(x_image, y_label, lr=0.1, n_steps=1, output_probs=False):
    
    original_image = x_image
    probs_per_step = []
    
    # Calculate loss, derivative and create adversarial image
    # https://www.tensorflow.org/versions/r0.11/api_docs/python/train/gradient_computation
    loss = tl.cost.cross_entropy(y, y_label, 'loss')
    deriv = tf.gradients(loss, x)
    image_adv = tf.stop_gradient(x - tf.sign(deriv)*lr/n_steps)
    image_adv = tf.clip_by_value(image_adv, 0, 1) # prevents -ve values creating 'real' image
    
    for _ in range(n_steps):
        # Calculate derivative and adversarial image
        dydx = sess.run(deriv, {x: x_image}) # can't seem to access 'deriv' w/o running this
        x_adv = sess.run(image_adv, {x: x_image})
        
        # Create darray of 3 images - orig, noise/delta, adversarial

        x_image = np.reshape(x_adv, (1, 28,28,1))
        
        img_adv_list = original_image
        img_adv_list = np.append(img_adv_list, dydx[0], axis=0)
        img_adv_list = np.append(img_adv_list, x_image, axis=0)
        test = x_image
        
        # Print/plot images and return probabilities
        probs = plot_predictions(img_adv_list, output_probs=output_probs, adversarial=True)
        probs_per_step.append(probs) if output_probs else None
    
    return probs_per_step, test




model_file_name = "./Save/mnist.ckpt"
resume = True # load model, resume from previous checkpoint?

X_train, y_train, _, _, X_test, y_test = \
    tl.files.load_mnist_dataset(shape=(-1, 28, 28, 1))

X_train = np.asarray(X_train, dtype=np.float32)
y_train = np.asarray(y_train, dtype=np.int32)
X_test = np.asarray(X_test, dtype=np.float32)
y_test = np.asarray(y_test, dtype=np.int32)




sess = tf.InteractiveSession()

#batch_size = 128

x = tf.placeholder(tf.float32, shape=[None, 28, 28, 1])   # [batch_size, height, width, channels]
y_ = tf.placeholder(tf.int64, shape=[None,])


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

y = network.outputs


cost = tl.cost.cross_entropy(y, y_, 'cost')

correct_prediction = tf.equal(tf.argmax(y, 1), y_)
acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
y_prob = tf.nn.softmax(y)

# train
n_epoch = 20
learning_rate = 0.0001
print_freq = 1

train_params = network.all_params
train_op = tf.train.AdamOptimizer(learning_rate, beta1=0.9, beta2=0.999,
    epsilon=1e-08, use_locking=False).minimize(cost, var_list=train_params)

tl.layers.initialize_global_variables(sess)

if resume:
    print('==============')
    print("Load existing model " + "!"*10)
    saver = tf.train.Saver()
    saver.restore(sess, model_file_name)
    print('==============')



#network.print_params()
#network.print_layers()
#print('==============')
#
#for epoch in range(n_epoch):
#    start_time = time.time()
#    for X_train_a, y_train_a in tl.iterate.minibatches(
#                                X_train, y_train, batch_size, shuffle=True):
#        feed_dict = {x: X_train_a, y_: y_train_a}
#        feed_dict.update( network.all_drop )        # enable noise layers
#        sess.run(train_op, feed_dict=feed_dict)
#
#    if epoch + 1 == 1 or (epoch + 1) % print_freq == 0:
#        print("Epoch %d of %d took %fs" % (epoch + 1, n_epoch, time.time() - start_time))
#        train_loss, train_acc, n_batch = 0, 0, 0
#        for X_train_a, y_train_a in tl.iterate.minibatches(
#                                X_train, y_train, batch_size, shuffle=True):
#            feed_dict = {x: X_train_a, y_: y_train_a}
#            err, ac = sess.run([cost, acc], feed_dict=feed_dict)
#            train_loss += err; train_acc += ac; n_batch += 1
#        print("   train loss: %f" % (train_loss/ n_batch))
#        print("   train acc: %f" % (train_acc/ n_batch))
#        test_loss, test_acc, n_batch = 0, 0, 0
#        for X_test_a, y_test_a in tl.iterate.minibatches(
#                                    X_test, y_test, batch_size, shuffle=True):
#            feed_dict = {x: X_test_a, y_: y_test_a}
#            err, ac = sess.run([cost, acc], feed_dict=feed_dict)
#            test_loss += err; test_acc += ac; n_batch += 1
#        print("   test loss: %f" % (test_loss/ n_batch))
#        print("   test acc: %f" % (test_acc/ n_batch))
#
#print("Save model " + "!"*10)
#saver = tf.train.Saver()
#save_path = saver.save(sess, model_file_name)
            
#sess.close()





if __name__ =='__main__':
#    print("test accuracy %g"%acc.eval(feed_dict={x: X_test[0:500],y_: y_test[0:500]}))
    pass
    
#    # plot_predictions
#    index_of_2s = [idx for idx, e in enumerate(y_test) if e==1][0:6]
#    x_batch = X_test[index_of_2s]
#    plot_predictions(x_batch)
    


    # Pick a random 2 image from first 1000 images 
    # Create adversarial image and with target label 6
    index_of_2s = [idx for idx, e in enumerate(y_test) if e==2][0:100]
    rand_index = np.random.randint(0, len(index_of_2s))
    image_norm = X_test[index_of_2s[rand_index]:index_of_2s[rand_index]+1]
#    image_norm = np.reshape(image_norm, (1, 28,28,1))
    label_adv = [6]#np.array([0,0,0,0,0,0,1,0,0,0]) # one hot encoded, adversarial label 6
    # Plot adversarial images
    # Over each step, model certainty changes from 2 to 6
    _, test= create_plot_adversarial_images(image_norm, label_adv, lr=0.2, n_steps=10)    

    

