#! /usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2017/11/30 19:34
# @Author  : Shiyu Li
# @Software: PyCharm


import random
import os
import shutil
import tensorflow as tf
from PIL import Image


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

def data_to_tfrecord(path, tfrecordname):
    '''
    :param path: path of data set
    :param tfrecordname: name of the tfrecord file


    data structure. 'cover' 'stego' is label/classes.

    cover --cover1.pgm
            cover2.pgm
            cover4.pgm
         ...
    stego --stego1.pgm
            stego2.pgm
            stego3.pgm
    ...
    '''
    classes = {'cover', 'stego'}
    writer = tf.python_io.TFRecordWriter(tfrecordname)
    for index, name in enumerate(classes):
        class_path = path + name +'\\'
        for img_name in os.listdir(class_path):
            print(name, end=' ')
            print(img_name)
            img_path = class_path + img_name
            img = Image.open(img_path)
            img = img.resize((256, 256))
            img_raw = img.tobytes()
            example = tf.train.Example(features=tf.train.Features(feature={
                        'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[index])),
                        'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
                    }))
            writer.write(example.SerializeToString())
    writer.close()

if __name__ == '__main__':
    # 将40000个cover和40000个stego中, 各抽出8000作为test, 剩下的作为train data 放在对应文件夹里

    ## 1. Corp data to train and test.

    # cover_data = '../database/CropedBoossBase/CroppedBossBase-1.0-256x256_cover'
    # stego_data = '../database/CropedBoossBase/CroppedBossBase-1.0-256x256_stego_SUniward0.4bpp'
    # root = '../database/CropedBoossBase'
    # makefile(root, cover_data, typename='cover', NUM=32000)
    # makefile(root, stego_data, typename='stego', NUM=32000)

    ## 2. Make TFrecords
    train_set_path = '../database/CropedBoossBase/train/'
    test_set_path = '../database/CropedBoossBase/test/'
    # data_to_tfrecord(train_set_path, 'train.tfrecords')
    data_to_tfrecord(test_set_path, 'test.tfrecords')