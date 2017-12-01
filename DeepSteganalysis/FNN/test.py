#! /usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2017/12/1 17:31
# @Author  : Shiyu Li
# @Software: PyCharm

'''
import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import *
import numpy as np
import time

def read_and_decode_without_distortion(filename):

    filename_queue = tf.train.string_input_producer([filename])
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    feature = tf.parse_single_example(serialized_example,
                                      features={
                                          'label':tf.FixedLenFeature([], tf.int64),
                                          'img_raw':tf.FixedLenFeature([], tf.string),
                                      })
    img = tf.decode_raw(feature['img_raw'], tf.float32)
    img = tf.reshape(img, [256,256,1])
    #img = tf.cast(img, tf.float32) * (1./255) - 0.5
    label = tf.cast(feature['label'], tf.int32)

    return img, label

x , y =read_and_decode_without_distortion('test.tfrecords')

print(type(x))
'''
# if __name__ == '__main__':
#     main()


import tensorflow as tf
import numpy as np

tfrecords_filename = 'test.tfrecords'
record_iterator = tf.python_io.tf_record_iterator(path=tfrecords_filename)

for string_record in record_iterator:
    # Parse the next example
    example = tf.train.Example()
    example.ParseFromString(string_record)

    # Get the features you stored (change to match your tfrecord writing code)

    #
    # label = int(example.features.feature['label']
    #                             .int64_list
    #                             .value[0])


    img_string = (example.features.feature['image_raw']
                                  .bytes_list
                                  .value[0])
    # Convert to a numpy array (change dtype to the datatype you stored)
    img_1d = np.fromstring(img_string, dtype=np.float32)
    # Print the image shape; does it match your expectations?
    # print(label)
    print('img_1d.shape')
    print(img_1d.shape)