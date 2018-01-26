#! /usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2018/1/22 14:50
# @Author  : Shiyu Li
# @Software: PyCharm

import math
import numpy as np
import cv2
from scipy import fftpack

def dis(img):
    cv2.namedWindow("Image") 
    cv2.imshow("Image", img)
    cv2.waitKey (0)
    cv2.destroyAllWindows()  

class JSteg:
    def __init__(self):
        self.img = None
        self.dct = None
        self.dct_quantified = None
        self.max_info_len = 0
        self.row = 0
        self.col = 0
        self.available_info_len = 0
        self.decode_img = None

    def set_img(self, input_img):
        self.img = cv2.imread(input_img, flags=0)
        row, col = self.img.shape
        if not( (row/8).is_integer() and (col/8).is_integer()):
            print('像素横纵要为8的倍数')
            self.dct = self.img[:int(row/8)*8, :int(col/8)*8].astype(np.float32)
            row, col = self.dct.shape
            print('已裁剪为%d * %d' % row, col)
        else:
            self.dct = self.img.astype(np.float32)
            
        self.row = row
        self.col = col
        for i in range(0, row, 8):
            for j in range(0, col, 8):
                # 也可以用np.lib.stride_tricks.as_strided(sudoku, shape=shape, strides=strides) 
                temp = self.dct[ i:i+8 , j:j+8 ]
                temp_ = temp.astype(np.float32)-128
                temp_dct = cv2.dct(temp_)
                self.dct[ i:i+8 , j:j+8 ] = temp_dct
                
    
    def quantify(self):
        qu_array = np.array([[16,11,10,16,1,1,1,1],
                            [12,12,14,1,1,1,1,55],
                            [14,13,1,1,1,1,69,56],
                            [14,1,1,1,1,87,80,62],
                            [1,1,1,1,68,109,109,77],
                            [1,1,1,64,81,104,104,92],
                            [1,1,78,87,103,121,121,101],
                            [1,92,95,98,112,100,100,99]])
        self.dct_quantified = self.dct
        for i in range(0, self.row, 8):
            for j in range(0, self.col, 8):
                temp = self.dct[ i:i+8 , j:j+8 ]
                #也可以直接舍去小数部分
                temp = np.around(temp/qu_array)
                self.dct[ i:i+8 , j:j+8 ] = temp
        
        self.available_info_len = (self.dct_quantified != 0).sum() + (self.dct_quantified !=1).sum()
        print('available_info_len = %d' % self.available_info_len)
                
    def write(self, pgm_info):
        info_len = self.col * self.row
        info_index = 0
        info = pgm_info.flatten()
        while True:
            if info_index >= info_len:
                break
            for i in range(self.row):
                for j in range(self.col):
                    if self._write((i,j), info[info_index]):
                        info_index += 1
        
        

    def _write(self, index, data):
        origin = self.dct_quantified[index]
        if origin in (-1, 1, 0):
            return False
        lower_bit = origin % 2
        if lower_bit == data:
            pass
        elif origin > 0:
            if (lower_bit, data) == (0, 1):
                self.dct_quantified[index] = origin + 1
            elif (lower_bit, data) == (1, 0):
                self.dct_quantified[index] = origin - 1
        elif origin < 0:
            if (lower_bit, data) == (0, 1):
                self.dct_quantified[index] = origin - 1
            elif (lower_bit, data) == (1, 0):
                self.dct_quantified[index] = origin + 1        
                

    def read(self, row,col):
        



'''
class Jsteg:
    def __init__(self):
        self.sequence_after_dct = None

    def set_sequence_after_dct(self, sequence_after_dct):
        self.sequence_after_dct = sequence_after_dct
        self.available_info_len = len([i for i in self.sequence_after_dct if i not in (-1, 1, 0)])  # 不是绝对可靠的
        print("Load>> 可嵌入", self.available_info_len, 'bits')

    def get_sequence_after_dct(self):
        return self.sequence_after_dct

    def write(self, info):
        """先嵌入信息的长度，然后嵌入信息"""
        info = self._set_info_len(info)
        info_len = len(info)
        info_index = 0
        im_index = 0
        while True:
            if info_index >= info_len:
                break
            data = info[info_index]
            if self._write(im_index, data):
                info_index += 1
            im_index += 1

    def read(self):
        """先读出信息的长度，然后读出信息"""
        _len, sequence_index = self._get_info_len()
        info = []
        info_index = 0

        while True:
            if info_index >= _len:
                break
            data = self._read(sequence_index)
            if data != None:
                info.append(data)
                info_index += 1
            sequence_index += 1

        return info

    # ===============================================================#

    def _set_info_len(self, info):
        l = int(math.log(self.available_info_len, 2)) + 1
        info_len = [0] * l
        _len = len(info)
        info_len[-len(bin(_len)) + 2:] = [int(i) for i in bin(_len)[2:]]
        return info_len + info

    def _get_info_len(self):
        l = int(math.log(self.available_info_len, 2)) + 1
        len_list = []
        _l_index = 0
        _seq_index = 0
        while True:
            if _l_index >= l:
                break
            _d = self._read(_seq_index)
            if _d != None:
                len_list.append(str(_d))
                _l_index += 1
            _seq_index += 1
        _len = ''.join(len_list)
        _len = int(_len, 2)
        return _len, _seq_index

    def _write(self, index, data):
        origin = self.sequence_after_dct[index]
        if origin in (-1, 1, 0):
            return False

        lower_bit = origin % 2
        if lower_bit == data:
            pass
        elif origin > 0:
            if (lower_bit, data) == (0, 1):
                self.sequence_after_dct[index] = origin + 1
            elif (lower_bit, data) == (1, 0):
                self.sequence_after_dct[index] = origin - 1
        elif origin < 0:
            if (lower_bit, data) == (0, 1):
                self.sequence_after_dct[index] = origin - 1
            elif (lower_bit, data) == (1, 0):
                self.sequence_after_dct[index] = origin + 1

        return True

    def _read(self, index):
        if self.sequence_after_dct[index] not in (-1, 1, 0):
            return self.sequence_after_dct[index] % 2
        else:
            return None


if __name__ == "__main__":
    jsteg = Jsteg()
    # 写
    sequence_after_dct = [-1, 0, 1] * 100 + [i for i in range(-7, 500)]
    jsteg.set_sequence_after_dct(sequence_after_dct)
    info1 = [0, 1, 0, 1, 1, 0, 1, 0]
    jsteg.write(info1)
    sequence_after_dct2 = jsteg.get_sequence_after_dct()
    # 读
    jsteg.set_sequence_after_dct(sequence_after_dct2)
    info2 = jsteg.read()
    print(info2)
'''