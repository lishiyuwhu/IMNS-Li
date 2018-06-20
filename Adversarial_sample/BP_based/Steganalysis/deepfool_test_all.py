#! /usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2018/3/14 12:58
# @Author  : Shiyu Li
# @Software: PyCharm

import JSteg
import tensorlayer as tl
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
from tensorlayer.layers import *
from mpl_toolkits.axes_grid1 import ImageGrid
import numpy as np
import os

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
    if image_list.max() < 2:
        temp = temp*255
    prob = y_prob.eval(feed_dict={x: temp})
    return prob
    
def Review_img(image_list, image_labels):
    fig = plt.figure(1, (5., 5.))
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                     nrows_ncols=(3, 3),  # creates 2x2 grid of axes
                     axes_pad=0.3,  # pad between axes in inch.
                     )

    for i in range(len(image_list)):
        image = image_list[i].reshape(256, 256)
        grid[i].imshow(image)
        grid[i].set_title('Label: {0}'.format(image_labels[i].argmax()))

    plt.show()


def plot_predictions(image_list, output_probs=False, adversarial=False):
    '''
    Evaluate images against trained model and plot images.
    If adversarial == True, replace middle image title appropriately
    Return probability list if output_probs == True
    '''
    # prob = y_prob.eval(feed_dict={x: image_list})
    prob = get_y_prob(image_list)
    pred_list = np.zeros(len(image_list)).astype(int)
    pct_list = np.zeros(len(image_list)).astype(int)

    # Setup image grid
    import math
    cols = 3
    rows = math.ceil(image_list.shape[0] / cols)
    fig = plt.figure(1, (12., 12.))
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                     nrows_ncols=(rows, cols),  # creates grid of axes
                     axes_pad=0.5,  # pad between axes in inch.
                     )

    # Get probs, images and populate grid
    for i in range(len(prob)):
        pred_list[i] = np.argmax(prob[i])
        pct_list[i] = prob.max(axis=1)[i] * 100
        image = image_list[i].reshape(256, 256)
        grid[i].imshow(image)

        grid[i].set_title('Label: {0} \nCertainty: {1}%' \
                          .format(pred_list[i],
                                  pct_list[i]))

        # Only use when plotting original, partial deriv and adversarial images
        if (adversarial) & (i % 3 == 1):
            grid[i].set_title("Adversarial \nPartial Derivatives")
    plt.show()

    return prob if output_probs else None


def create_plot_adversarial_images(x_image, y_label, lr=0.1, n_steps=1, output_probs=False):
    
    original_image = x_image
    probs_per_step = []
    
    # Calculate loss, derivative and create adversarial image
    # https://www.tensorflow.org/versions/r0.11/api_docs/python/train/gradient_computation
    loss = tl.cost.cross_entropy(y, y_label, 'loss')
    deriv = tf.gradients(loss, x)

    eta = tf.div(tf.multiply(deriv, tf.reduce_max(y_prob)), tf.square(tf.norm(deriv,ord=2)))

    image_adv = tf.stop_gradient(x - eta)
    image_adv = tf.clip_by_value(image_adv, 0, 1)
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


model_file_name = "Own_Steganography_Bossbase.npz"
resume = True  # load model, resume from previous checkpoint?

sess = tf.InteractiveSession()

# batch_size = 128


# ====================
x = tf.placeholder(tf.float32, shape=[None, 256, 256, 1])  # [batch_size, height, width, channels]
y_ = tf.placeholder(tf.int32, shape=[None, ])

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
cost = tl.cost.cross_entropy(y, y_, name='cost')

correct_prediction = tf.equal(tf.cast(tf.argmax(y, 1), tf.int32), y_)
acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
y_prob = tf.nn.softmax(y)

# =================
tl.layers.initialize_global_variables(sess)

if resume:
    print("Load existing model " + "!" * 10)
    tl.files.load_and_assign_npz(sess=sess, name=model_file_name, network=net)

if __name__ == '__main__':
    '''
    target = JSteg.JSteg()
    target.set_img('7.pgm')
    target.write('42_85p85.pgm')
    
    raw = cv2.imread('7.pgm', 0).astype(np.float32)
    line = 90
    raw[:line]= target.encode_img[:line]
    target.read(85,85,raw)
    JSteg.dis(target.decode_img)

    
    encode_image_temp = target.encode_img.copy()
    encode_image = encode_image_temp/255.

    print("========================")
    print("encode_img的预测结果")
    if len(encode_image.shape)==2:
        encode_image.resize([1,256,256,1])
    plot_predictions(encode_image*255)

    print("========================")
    print("encode_img_adv的预测结果")
    _, encode_image_adv = create_plot_adversarial_images\
                            (encode_image, [0], lr=0.002, n_steps=5)

    #encode_image  encode_image_adv拼接
    print("========================")
    print("encode_image  encode_image_adv拼接, 前96行")
    if len(encode_image_adv.shape)==2 and len(encode_image.shape)==2:
        encode_image_adv[:96] = encode_image[:96]
        print('shape=2')
    if len(encode_image_adv.shape)==4 and len(encode_image.shape)==4:
        encode_image_adv[0][:96] = encode_image[0][:96]
        print('shape=4')




    print("========================")
    print("拼接后的encode_img_adv的预测结果")
    plot_predictions(encode_image_adv)

    print("========================")
    print("decode")
    target.read(85,85,encode_image_adv.reshape([256,256])*255)
    plt.imshow(target.decode_img)
    '''



    #==================================


    lr = 0.0003
    num = 300
    psnr_total = 0
    test_path = 'cover'
    encode_imglist = np.array([])
#    for img_name in os.listdir(test_path):
#        img_path = 'cover\\' + img_name
#        print(img_path)
#        temp = JSteg.JSteg()
#        temp.set_img(img_path)
#        temp.write('42_85p85.pgm')
#        encode_image_temp = temp.encode_img.copy()
#        encode_image_temp.resize([1,256,256,1])
#        encode_image = encode_image_temp/255.  
#        _, encode_image_adv = create_plot_adversarial_images\
#                        (encode_image, [0], lr=lr, n_steps=2)
#        temp = encode_image_adv.copy()
#        encode_image_adv[0][:128] = encode_image[0][:128] 
#        if len(encode_imglist)==0:
#            encode_imglist = encode_image_adv.copy()
#        else:
#            encode_imglist = np.concatenate((encode_imglist,encode_image_adv), axis=0)
#        
#        psnr_total += psnr(encode_image, temp.reshape([256,256]))
    
    a = np.load("encode_img.npy")/255.

    for img in a:
        temp = img.reshape([1,256,256,1])
        _, encode_image_adv = create_plot_adversarial_images\
                            (temp, [0], lr=lr, n_steps=2)    
        if len(encode_imglist)==0:
            encode_imglist = encode_image_adv.copy()
        else:
            encode_imglist = np.concatenate((encode_imglist,encode_image_adv), axis=0)
    print(encode_imglist.shape)

    psnr_avg = psnr_total/num
    print('psnr_avg', psnr_avg)
    
    lable = np.asarray([1]*301, dtype=np.int32)
    acc1 = sess.run(acc, feed_dict={x:encode_imglist, y_:lable})
    print('acc', acc1)

    prob_sum = 0
    for a in get_y_prob(encode_imglist):
        prob_sum += a[0]
    print('prob_avg', prob_sum/num)
    
    print('lr', lr)
#    a = encode_imglist[6]
#    temp.read(85,85,a.reshape(256,256)*255)
#    plt.imshow(temp.decode_img)    
