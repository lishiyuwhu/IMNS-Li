#! /usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2018/1/11 9:58
# @Author  : Shiyu Li
# @Software: PyCharm

import tensorflow as tf
import numpy as np


def read_and_decode(filename, is_train=None):
    """ Return tensor to read from TFRecord """
    filename_queue = tf.train.string_input_producer([filename])
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'label': tf.FixedLenFeature([], tf.int64),
                                           'img_raw' : tf.FixedLenFeature([], tf.string),
                                       })
    # You can do more image distortion here for training data
    img = tf.decode_raw(features['img_raw'], tf.float32)
    img = tf.reshape(img, [32, 32, 3])
    # img = tf.cast(img, tf.float32) #* (1. / 255) - 0.5
    if is_train == True:
        # 1. Randomly crop a [height, width] section of the image.
        img = tf.random_crop(img, [24, 24, 3])
        # 2. Randomly flip the image horizontally.
        img = tf.image.random_flip_left_right(img)
        # 3. Randomly change brightness.
        img = tf.image.random_brightness(img, max_delta=63)
        # 4. Randomly change contrast.
        img = tf.image.random_contrast(img, lower=0.2, upper=1.8)
        # 5. Subtract off the mean and divide by the variance of the pixels.
        try: # TF 0.12+
            img = tf.image.per_image_standardization(img)
        except: # earlier TF versions
            img = tf.image.per_image_whitening(img)

    elif is_train == False:
        # 1. Crop the central [height, width] of the image.
        img = tf.image.resize_image_with_crop_or_pad(img, 24, 24)
        # 2. Subtract off the mean and divide by the variance of the pixels.
        try: # TF 0.12+
            img = tf.image.per_image_standardization(img)
        except: # earlier TF versions
            img = tf.image.per_image_whitening(img)
    elif is_train == None:
        img = img

    label = tf.cast(features['label'], tf.int32)
    return img, label




# if __name__ == '__main__':
#     main()