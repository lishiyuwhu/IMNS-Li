#! /usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2018/3/6 15:33
# @Author  : Shiyu Li
# @Software: PyCharm

import numpy as np
import tensorlayer as tl
import tensorflow as tf
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

def perturb_image(xs, img):
    # If this function is passed just one perturbation vector,
    # pack it in a list to keep the computation the same
    if img.shape == (1,28,28,1):
        img.resize([28,28])

    if xs.ndim < 2:
        xs = np.array([xs])

    # Copy the image n == len(xs) times so that we can
    # create n new perturbed images
    tile = [len(xs)] + [1] * (xs.ndim)
    imgs = np.tile(img, tile)

    # Make sure to floor the members of xs as int types
    xs = xs.astype(int)

    for x, img in zip(xs, imgs):
        # Split x into an array of 5-tuples (perturbation pixels)
        # i.e., [[x,y,r,g,b], ...]
        pixels = np.split(x, len(x) // 3)
        for pixel in pixels:
            # At each pixel's x,y position, assign its rgb value

            x_pos, y_pos, rgb = pixel
            img[x_pos, y_pos] = rgb
            img.resize([1,28,28,1])

    return imgs

def predict_classes(xs, img, target_class=None, minimize=True):
    # Perturb the image with the given pixel(s) x and get the prediction of the model
    imgs_perturbed = perturb_image(xs, img)
    if imgs_perturbed.shape != (1,28,28,1):
        imgs_perturbed.resize([1,28,28,1])
    
    prob = y_prob.eval(feed_dict={x: imgs_perturbed})
    predictions = prob.max(axis=1)
    # This function should always be minimized, so return its complement if needed
    return predictions if minimize else 1 - predictions

def Review_img(image_list, image_labels = None):
    '''
    :param image_list: [img1, img2, ...]
    :param image_labels: [[one_hot1], [one_hot2], ...]
    :return: None
    '''
    fig = plt.figure(1, (5., 5.))
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                     nrows_ncols=(3, 3),  # creates 2x2 grid of axes
                     axes_pad=0.3,  # pad between axes in inch.
                     )

    for i in range(len(image_list)):
        image = image_list[i].reshape(28, 28)
        grid[i].imshow(image)
        if image_labels!=None:
            grid[i].set_title('Label: {0}'.format(image_labels[i].argmax()))
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
    rows = math.ceil(image_list.shape[0] / cols)
    fig = plt.figure(1, (12., 12.))
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                     nrows_ncols=(rows, cols),  # creates grid of axes
                     axes_pad=0.5,  # pad between axes in inch.
                     )

    # Get probs, images and populate grid
    for i in range(len(prob)):
        pred_list[i] = np.argmax(prob[i])  # for mnist index == classification
        pct_list[i] = prob.max(axis=1)[i] * 100

        image = image_list[i].reshape(28, 28)
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

def attack_success(x, img, target_class, model, targeted_attack=False, verbose=False):
    # Perturb the image with the given pixel(s) and get the prediction of the model
    attack_image = perturb_image(x, x_test[img])
    if attack_image.shape != (1,28,28,1):
        attack_image.resize([1,28,28,1])
    confidence = y.eval(feed_dict={x: attack_image})
    predicted_class = np.argmax(confidence)
    
    # If the prediction is what we want (misclassification or 
    # targeted classification), return True
    if (verbose):
        print('Confidence:', confidence[target_class])
    if ((targeted_attack and predicted_class == target_class) or
        (not targeted_attack and predicted_class != target_class)):
        return True



model_file_name = "model.npz"
resume = True  # load model, resume from previous checkpoint?


#====================
sess = tf.InteractiveSession()

_, _, _, _, X_test, y_test = \
    tl.files.load_mnist_dataset(shape=(-1, 28, 28, 1))
X_test = np.asarray(X_test, dtype=np.float32)
y_test = np.asarray(y_test, dtype=np.int32)

# train
# batch_size = 128
n_epoch = 5
learning_rate = 0.0001
print_freq = 1

sess = tf.InteractiveSession()
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
    pass
    img_index = 42
    img_norm = X_test[img_index].reshape([1,28,28,1])
    

    # test Review_img() plot_predictions()
    # Review_img([X_test[img_index]], [np.array([0,0,0,0,1,0,0,0,0,0])])
    # plot_predictions(img_norm)
    
    # test perturb_image()
    # pixel = np.array([5, 5, 1])
    # img_pre = perturb_image(pixel, img_norm)
    # img_pre.resize([1,28,28,1])
    # print(predict_classes(pixel, img_pre))