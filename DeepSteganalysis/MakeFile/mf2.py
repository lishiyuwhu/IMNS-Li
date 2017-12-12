#! /usr/bin/python
# -*- coding: utf8 -*-


import tensorflow as tf
import tensorlayer as tl
import numpy as np
from PIL import Image
import io
import os



## Save data ==================================================================
classes = ['/data/0cover', '/data/1stego']  # cat is 0, dog is 1
cwd = os.getcwd()
writer = tf.python_io.TFRecordWriter("train.tfrecords")
for index, name in enumerate(classes):
    class_path = cwd + name + "/"
    for img_name in os.listdir(class_path):
        img_path = class_path + img_name
        img = Image.open(img_path)
        img = img.resize((256,256))
        img_raw = img.tobytes()
        example = tf.train.Example(features=tf.train.Features(feature={ # SequenceExample for seuqnce example
            "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[index])),
            'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw])),
        }))
        writer.write(example.SerializeToString())  # Serialize To String
writer.close()