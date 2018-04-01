#! /usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2018/3/10 15:48
# @Author  : Shiyu Li
# @Software: PyCharm
# Python Libraries


import numpy as np
import matplotlib
import tensorlayer as tl
import tensorflow as tf
from tensorlayer.layers import *
from matplotlib import pyplot as plt
from differential_evolution import differential_evolution
import cv2, os
import JSteg

def psnr(im1,im2):
    
    if im1.max() < 2:
        mmax = 1
    else:
        mmax = 255
    diff = np.abs(im1 - im2)
    diff = diff.flatten()
    mse = sum(x*x for x in diff)
    MSE = mse/(len(diff))
    psnr = 10* np.log10(mmax*mmax/MSE)
    return psnr

def get_y_prob(image_list):
    temp = image_list.copy()
    if temp.max() < 2:
        temp = temp*255
    if len(temp.shape)<4:
        temp.resize([1,256,256,1])
    prob = y_prob.eval(feed_dict={x: temp})
    return prob


def plot_image(image, label_true=None, class_names=None, label_pred=None):
    '''

    :param image: image.shape=[28,28]
    :param label_true:
    :param class_names:
    :param label_pred: 
    :return:
    '''
    if image.shape != (256,256):
        image = image.reshape([256,256])
    
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
    img_temp = img.copy()
    # If this function is passed just one perturbation vector,
    # pack it in a list to keep the computation the same
    if xs.ndim < 2:
        xs = np.array([xs])
    xs = xs.astype(np.uint8)
    # Copy the image n == len(xs) times so that we can 
    # create n new perturbed images
    tile = [len(xs)] + [1]*(xs.ndim+1)
    imgs = np.tile(img_temp, tile)
    
    # Make sure to floor the members of xs as int types
    xs = xs.astype(int)
    
    for x,img_t in zip(xs, imgs):
        # Split x into an array of 5-tuples (perturbation pixels)
        # i.e., [[x,y,r,g,b], ...]
        pixels = np.split(x, len(x) // 3)
        for pixel in pixels:
            # At each pixel's x,y position, assign its rgb value
            x_pos, y_pos, *rgb = pixel
            img_t[x_pos, y_pos] = rgb
    
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
    prob = get_y_prob(imgs_perturbed)#y_prob.eval(feed_dict={x: imgs_perturbed})

    
    
    predictions = prob[:,target_class]
    # This function should always be minimized, so return its complement if needed
    return predictions if minimize else 1 - predictions

def attack_success(x_1, img, target_class, targeted_attack=False, verbose=False):
    # Perturb the image with the given pixel(s) and get the prediction of the model
    attack_image = perturb_image(x_1, X_test[img])

#    confidence = y_prob.eval(feed_dict={x: attack_image})[0]
    confidence = get_y_prob(attack_image)[0]
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
    bounds = [(88,255), (0,255), (0,255)] * pixel_count
    
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
    
#    prior_probs = y_prob.eval(feed_dict={x: X_test[img].reshape([1,256,256,1])})[0]
#    predicted_probs = y_prob.eval(feed_dict={x: attack_image})[0]
    
    prior_probs = get_y_prob(X_test[img])[0]
    predicted_probs= get_y_prob(attack_image)[0]
    
    
    
    predicted_class = np.argmax(predicted_probs)
    actual_class = y_test[img]
    success = predicted_class != actual_class
    cdiff = prior_probs[actual_class] - predicted_probs[actual_class]

    # Show the best attempt at a solution (successful or not)
    plot_image(attack_image, actual_class, class_names, predicted_class)

    return [ attack_image, pixel_count, img, actual_class, predicted_class, success, cdiff, prior_probs, predicted_probs, attack_result.x]
# ====================
model_file_name = "Own_Steganography_Bossbase.npz"
resume = True  # load model, resume from previous checkpoint?
class_names = ['cover', 'stego']

sess = tf.InteractiveSession()

# train
# batch_size = 128
n_epoch = 5
learning_rate = 0.0001
print_freq = 1

sess = tf.InteractiveSession()
x = tf.placeholder(tf.float32, shape=[None, 256, 256, 1])  # [batch_size, height, width, channels]
y_ = tf.placeholder(tf.int64, shape=[None, ])
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

net = InputLayer(x, name='inputlayer')
net = Conv2d(net, 1, (5, 5), (1, 1), act=tf.identity,
             padding='VALID', W_init=high_pass_filter, name='HighPass')
net = Conv2d(net, 64, (5, 5), (2, 2), act=tf.nn.relu,
             padding='VALID', W_init=W_init, name='trainCONV1')
net = Conv2d(net, 16, (5, 5), (2, 2), act=tf.nn.relu,
             padding='VALID', W_init=W_init, name='trainCONV2')
net = FlattenLayer(net, name='trainFlatten')
net = DenseLayer(net, n_units=500, act=tf.nn.relu,
                 W_init=W_init2, b_init=b_init2, name='trainFC1')
net = DenseLayer(net, n_units=500, act=tf.nn.relu,
                 W_init=W_init2, b_init=b_init2, name='trainFC2')
net = DenseLayer(net, n_units=2, act=tf.identity,
                 W_init=W_init, name='trainOutput')
y = net.outputs
y_prob = tf.nn.softmax(y)

cost = tl.cost.cross_entropy(y, y_, 'cost')
correct_prediction = tf.equal(tf.argmax(y, 1), y_)
acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

train_params = net.all_params
train_op = tf.train.AdamOptimizer(learning_rate, beta1=0.9, beta2=0.999,
                                  epsilon=1e-08, use_locking=False).minimize(cost, var_list=train_params)
tl.layers.initialize_global_variables(sess)

if resume:
    print("Load existing model " + "!" * 10)
    tl.files.load_and_assign_npz(sess=sess, name=model_file_name, network=net)
    


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
    # image = 42 # num of the image in X_test
    # image_norm = X_test[image]
    
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
    
    # pixels = 3 # Number of pixels to attack
    # img_adv,*_ = attack(image, pixel_count=pixels, verbose=True)

    

    test_path = 'cover'
    filenum = 5
    X_test = []
    y_test= [1]*filenum
    pixel_num = 80
    for img_name in os.listdir(test_path):
        img_path = 'cover\\' + img_name
        target = JSteg.JSteg()  
        target.set_img(img_path)
        img_norm = cv2.imread(img_path,0)
        target.write('42_85p85.pgm')
        temp = target.encode_img.reshape([1,256,256,1])
        if len(X_test) ==0 :
            X_test=temp
        X_test= np.concatenate((X_test, temp), axis=0)
        
    def do(img_index,pixel_count):
        img_adv,*_,  predicted_probs,_ = attack(img_index, pixel_count=pixel_count, verbose=True)
        psnr_ = psnr(img_adv.reshape([256,256]), X_test[i].reshape([256,256]))
        suc = 0
        if predicted_probs[0] >0.5:
            suc = 1
        return [suc, predicted_probs[0] ,psnr_]
    
    i=1
    
    Prob=0
    Psnr = 0
    Succ = 0
    for i in range(filenum):
        for j in range(10):
            success, prob_, psnr_ = do(i, pixel_num)
            if success==1:
                Psnr += psnr_
                Prob += prob_
    
    print('Psnr_avg', Psnr/filenum)
    print('Prob_avg', Prob/filenum)
        
        

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
#    target = JSteg.JSteg()
#    target.set_img('7.pgm')
#    img_norm = cv2.imread('7.pgm',0)
#    target.write('42_85p85.pgm')
#    X_test = target.encode_img.reshape([1,256,256,1])
#    y_test = [1]
#    print(y_prob.eval(feed_dict={x: target.encode_img.reshape([1,256,256,1])}))
#    img_index = 0
#    pixel_count=100
#    
#    def do():
#        img_adv,*_,cdiff, _,pixel_perb = attack(img_index, pixel_count=pixel_count, verbose=True)
#        print(' pixel_count=',pixel_count )
#        target.read(85,85,img_adv.reshape([256,256]))
#        plt.imshow(target.decode_img)
#        return img_adv
#    i=1
#    while True:
#        img_adv = do()
#        confi = get_y_prob(img_adv)[0][1]
#        if confi < 0.5:
#            break
#        i += 1
#        
#    print('这是第', i,'次')
#    img_adv = do()    
#    print('第', i,'次')
#    print(' pixel_count=',pixel_count )
#    target.read(85,85,img_adv.reshape([256,256]))
#    plt.imshow(target.decode_img)

