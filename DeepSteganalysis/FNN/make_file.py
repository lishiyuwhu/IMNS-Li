#! /usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2017/11/30 19:34
# @Author  : Shiyu Li
# @Software: PyCharm


import random
import os
import shutil



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
        os.makedirs(root + '/train/' + typename)
    except FileExistsError:
        print('=================================================')
        print("There exist '/train' '/test', delete first please.")



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
            shutil.copyfile(os.path.join(data, name), os.path.join(root + '/test', typename + name))


if __name__ == '__main__':
    # 将40000个cover和40000个stego中, 各抽出8000作为test, 剩下的作为train data 放在对应文件夹里

    cover_data = '../database/CropedBoossBase/CroppedBossBase-1.0-256x256_cover'
    stego_data = '../database/CropedBoossBase/CroppedBossBase-1.0-256x256_stego_SUniward0.4bpp'
    root = '../database/CropedBoossBase'
    makefile(root, cover_data, typename='cover', NUM=32000)
    makefile(root, stego_data, typename= 'stego', NUM=32000)