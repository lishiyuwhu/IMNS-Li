#! /usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2017/12/12 20:08
# @Author  : Shiyu Li
# @Software: PyCharm
#
import tensorflow as tf
import os
from PIL import Image
import random, shutil

def makefile(root, data, typename, NUM):
    '''
    :param root: target file of '/train' and '/test'
    :param data: the data file you want to corp
    :param typename: data name
    :param NUM: the num of data in '/train'
    :return: None
    '''

    try:
        os.makedirs(root + '/train')
        os.makedirs(root + '/test')

    except BaseException:
        print('=================================================')
        print("There exist '/train' '/test', delete first please.")

    try:
        os.makedirs(root + '/train/' + typename)
        os.makedirs(root + '/test/' + typename)
    except BaseException:
        print("There exist '/%s', delete first please." % typename)


    name_list = os.listdir(data)
    random.shuffle(name_list)

    count = 0
    for name in name_list:
        if count < NUM:
            print('train data :%s' % name)
            shutil.copyfile(os.path.join(data, name), os.path.join(root + '/train/' + typename + '/', typename + name))
            count += 1
        else:
            print('test data :%s' % name)
            shutil.copyfile(os.path.join(data, name), os.path.join(root + '/test/'+ typename + '/', typename + name))

def encode(filename):
    '''
    file structure
    -img-to-TR.py
    -train
      -cover
        -1.pgm
        -3.pgm
      -stego
        -2.pgm
        -6.pgm
    -test
      -cover
        -5.pgm
        -8.pgm
      -stego
        -213.pgm
        -223.pgm

    :param filename:
    :return:
    '''
    cwd = os.getcwd()
    root = cwd + '\\' + filename + '\\'
    name = filename + '_Mnist_pgm.tfrecords'
    print(name)
    writer = tf.python_io.TFRecordWriter(name)
    classes = ['0','1','2','3','4','5','6','7','8','9']

    #for image_filename, label in zip(image_filename_list, label_list):

    for index, name in enumerate(classes):
        class_path = root + name + '\\'
        for img_name in os.listdir(class_path):
            img_path = class_path + img_name
            img = Image.open(img_path)
            print(img_name)
            data = img.tobytes()
            example = tf.train.Example(features=tf.train.Features(feature={
                "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[index])),
                'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[data]))
            }))
            writer.write(example.SerializeToString())
    writer.close()

if __name__ == '__main__':

    cover_data = 'CroppedBossBase-1.0-256x256_cover'
    stego_data = 'CroppedBossBase-1.0-256x256_stego_SUniward0.4bpp'
    root = os.getcwd()
    # makefile(root, cover_data, typename='cover', NUM=32000)
    # makefile(root, stego_data, typename='stego', NUM=32000)




    encode('train')
    encode('test')