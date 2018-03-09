#! /usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2018/3/9 10:44
# @Author  : Shiyu Li
# @Software: PyCharm

import tensorflow as tf
import cv2
import numpy as np
import JSteg
import os

def encode(img_name, img_info_name):
    temp = JSteg.JSteg()
    temp.set_img(img_name)
    temp.write(img_info_name)
    return temp.encode_img


def to_TR():
    img_info_name = '42_85p85.pgm'
    cwd = os.getcwd()
    cover_root = cwd + '\\cover\\'
    name = 'own_CroppedBossBase-1.0-256x256.tfrecords'
    writer = tf.python_io.TFRecordWriter(name)
    classes = ['cover', 'stego']
    
    
        # then for cover img
    for img_name in os.listdir(cover_root):
        img_path = cover_root + img_name
        img = cv2.imread(img_path,0)
        img = img.astype(np.float32)
        print(img.dtype)
        data = img.tobytes()
        example = tf.train.Example(features=tf.train.Features(feature={
            "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[0])),
            'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[data]))
        }))
        writer.write(example.SerializeToString())
    # first for stego img
    for img_name in os.listdir(cover_root):
        img_path = cover_root + img_name
        print(img_name)
        temp = encode(img_path, img_info_name=img_info_name)
        data = temp.tobytes()
        example = tf.train.Example(features=tf.train.Features(feature={
            "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[1])),
            'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[data]))
        }))
        writer.write(example.SerializeToString())



    writer.close()

if __name__ == '__main__':
    pass
    to_TR()
#    a = encode('7.pgm','42_85p85.pgm')
#    b=  JSteg.JSteg()
#    b.read(85,85,a)
#    JSteg.dis(b.decode_img)