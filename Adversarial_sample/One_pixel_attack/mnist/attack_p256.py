#! /usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2018/3/10 15:48
# @Author  : Shiyu Li
# @Software: PyCharm
# Python Libraries

import pickle
import numpy as np
import pandas as pd
import matplotlib
import tensorlayer as tl
import tensorflow as tf
from matplotlib import pyplot as plt
import requests
from differential_evolution import differential_evolution


def plot_image(image, label_true=None, class_names=None, label_pred=None):
    '''

    :param image: image.shape=[28,28]
    :param label_true:
    :param class_names:
    :param label_pred: 
    :return:
    '''
    if image.shape != (28,28):
        image = image.reshape([28,28])
    
    plt.grid()
    plt.imshow(image)
    # Show true and predicted classes
    if label_true is not None and class_names is not None:
        labels_true_name = class_names[label_true]
        if label_pred is None:
            xlabel = "True: " + labels_true_name
        else:
            # Name of the predicted class
            labels_pred_name = class_names[label_pred]

            xlabel = "True: " + labels_true_name + "\nPredicted: " + labels_pred_name

        # Show the class on the x-axis
        plt.xlabel(xlabel)

    plt.xticks([])  # Remove ticks from the plot
    plt.yticks([])
    plt.show()  # Show the plot

def perturb_image(xs, img):
    '''
    
    :param xs: np.array([5, 5, 1])
    :param img: X_test[image] . NOT A INDEX
    :return:[img_per1, img_per2, ...] (None, 28,28,1)
    '''
    # If this function is passed just one perturbation vector,
    # pack it in a list to keep the computation the same
    if xs.ndim < 2:
        xs = np.array([xs])
    
    # Copy the image n == len(xs) times so that we can 
    # create n new perturbed images
    tile = [len(xs)] + [1]*(xs.ndim+1)
    imgs = np.tile(img, tile)
    
    # Make sure to floor the members of xs as int types
    xs = xs.astype(int)
    
    for x,img in zip(xs, imgs):
        # Split x into an array of 5-tuples (perturbation pixels)
        # i.e., [[x,y,r,g,b], ...]
        pixels = np.split(x, len(x) // 3)
        for pixel in pixels:
            # At each pixel's x,y position, assign its rgb value
            x_pos, y_pos, *rgb = pixel
            img[x_pos, y_pos] = rgb
    
    return imgs

def predict_classes(xs, img, target_class, minimize=True):
    '''

    :param xs: 
    :param img: 
    :param target_class:
    :param minimize:
    :return:
    '''
    # Perturb the image with the given pixel(s) x and get the prediction of the model
    imgs_perturbed = perturb_image(xs, img)
    prob = y_prob.eval(feed_dict={x: imgs_perturbed})
    predictions = prob[:,target_class]
    # This function should always be minimized, so return its complement if needed
    return predictions if minimize else 1 - predictions

def attack_success(x_1, img, target_class, targeted_attack=False, verbose=False):
    # Perturb the image with the given pixel(s) and get the prediction of the model
    attack_image = perturb_image(x_1, X_test[img])

    confidence = y_prob.eval(feed_dict={x: attack_image})[0]
    
    predicted_class = np.argmax(confidence)
    
    # If the prediction is what we want (misclassification or 
    # targeted classification), return True
    if (verbose):
        print('Confidence:', confidence[target_class])
    if ((targeted_attack and predicted_class == target_class) or
        (not targeted_attack and predicted_class != target_class)):
        return True
    

def attack(img, target=None, pixel_count=1, 
           maxiter=75, popsize=400, verbose=False):
    # Change the target class based on whether this is a targeted attack or not
    targeted_attack = target is not None
    target_class = target if targeted_attack else y_test[img]
    
    # Define bounds for a flat vector of x,y,r,g,b values
    # For more pixels, repeat this layout
    bounds = [(0,28), (0,28), (0,256)] * pixel_count
    
    # Population multiplier, in terms of the size of the perturbation vector x
    popmul = max(1, popsize // len(bounds))
    
    # Format the predict/callback functions for the differential evolution algorithm
    predict_fn = lambda xs: predict_classes(
        xs, X_test[img], target_class, target is None)
    callback_fn = lambda x, convergence: attack_success(
        x, img, target_class, targeted_attack, verbose)
    
    # Call Scipy's Implementation of Differential Evolution
    attack_result = differential_evolution(
        predict_fn, bounds, maxiter=maxiter, popsize=popmul,
        recombination=1, atol=-1, callback=callback_fn, polish=False)

    # Calculate some useful statistics to return from this function
    attack_image = perturb_image(attack_result.x, X_test[img])
    
    prior_probs = y_prob.eval(feed_dict={x: X_test[image].reshape([1,28,28,1])})[0]
    predicted_probs = y_prob.eval(feed_dict={x: attack_image})[0]
    
    predicted_class = np.argmax(predicted_probs)
    actual_class = y_test[img]
    success = predicted_class != actual_class
    cdiff = prior_probs[actual_class] - predicted_probs[actual_class]

    # Show the best attempt at a solution (successful or not)
    plot_image(attack_image, actual_class, class_names, predicted_class)

    return [ attack_image, pixel_count, img, actual_class, predicted_class, success, cdiff, prior_probs, predicted_probs, attack_result.x]
# ====================
model_file_name = "model.npz"
resume = True  # load model, resume from previous checkpoint?
class_names = ['0','1','2','3','4','5','6','7','8','9']

sess = tf.InteractiveSession()
_, _, _, _, X_test, y_test = \
    tl.files.load_mnist_dataset(shape=(-1, 28, 28, 1))
X_test = np.asarray(X_test, dtype=np.float32)*256
y_test = np.asarray(y_test, dtype=np.int32)

# train
# batch_size = 128
n_epoch = 5
learning_rate = 0.0001
print_freq = 1

sess = tf.InteractiveSession()
x = tf.placeholder(tf.float32, shape=[None, 28, 28, 1])  # [batch_size, height, width, channels]
y_ = tf.placeholder(tf.int64, shape=[None, ])
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
y_prob = tf.nn.softmax(y)

cost = tl.cost.cross_entropy(y, y_, 'cost')
correct_prediction = tf.equal(tf.argmax(y, 1), y_)
acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

train_params = network.all_params
train_op = tf.train.AdamOptimizer(learning_rate, beta1=0.9, beta2=0.999,
                                  epsilon=1e-08, use_locking=False).minimize(cost, var_list=train_params)
tl.layers.initialize_global_variables(sess)

if resume:
    print("Load existing model " + "!" * 10)
    tl.files.load_and_assign_npz(sess=sess, name=model_file_name, network=network)
    


'''
for epoch in range(n_epoch):
    start_time = time.time()
    for X_train_a, y_train_a in tl.iterate.minibatches(
                                X_train, y_train, batch_size, shuffle=True):
        feed_dict = {x: X_train_a, y_: y_train_a}
        feed_dict.update( network.all_drop )        # enable noise layers
        sess.run(train_op, feed_dict=feed_dict)

    if epoch + 1 == 1 or (epoch + 1) % print_freq == 0:
        print("Epoch %d of %d took %fs" % (epoch + 1, n_epoch, time.time() - start_time))
        train_loss, train_acc, n_batch = 0, 0, 0
        for X_train_a, y_train_a in tl.iterate.minibatches(
                                X_train, y_train, batch_size, shuffle=True):
            feed_dict = {x: X_train_a, y_: y_train_a}
            err, ac = sess.run([cost, acc], feed_dict=feed_dict)
            train_loss += err; train_acc += ac; n_batch += 1
        print("   train loss: %f" % (train_loss/ n_batch))
        print("   train acc: %f" % (train_acc/ n_batch))
        test_loss, test_acc, n_batch = 0, 0, 0
        for X_test_a, y_test_a in tl.iterate.minibatches(
                                    X_test, y_test, batch_size, shuffle=True):
            feed_dict = {x: X_test_a, y_: y_test_a}
            err, ac = sess.run([cost, acc], feed_dict=feed_dict)
            test_loss += err; test_acc += ac; n_batch += 1
        print("   test loss: %f" % (test_loss/ n_batch))
        print("   test acc: %f" % (test_acc/ n_batch))

print("Save model " + "!"*10)
tl.files.save_npz(network.all_params , name=model_file_name, sess=sess)
'''

if __name__ == '__main__':
    image = 42 # num of the image in X_test
    image_norm = X_test[image]
    
    # plot_image(X_test[image].reshape([28,28]))
    
    # pixel = np.array([5, 5, 1]) # pixel = x,y,l
    # image_perturbed = perturb_image(pixel, X_test[image])[0]
    # plot_image(image_perturbed)

    # pixel = np.array([5, 5, 1])  # pixel = x,y,l
    # true_class = y_test[image]
    # prior_confidence = y_prob.eval(feed_dict={x: X_test[image].reshape([1,28,28,1])})[0,true_class]
    # confidence = predict_classes(pixel, X_test[image], true_class)[0]
    # success = attack_success(pixel, image, true_class, verbose=True)
    # print('Confidence in true class', true_class, 'is', confidence)
    # print('Prior confidence was', prior_confidence)
    # print('Attack success:', success == True)
    # plot_image(perturb_image(pixel, X_test[image])[0])
    
    pixels = 3 # Number of pixels to attack

    img_adv,*_ = attack(image, pixel_count=pixels, verbose=True)

