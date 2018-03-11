#! /usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2018/1/22 14:50
# @Author  : Shiyu Li
# @Software: PyCharm

import numpy as np
import cv2


def dis(img):
    img2 = img
    if img.dtype != 'uint8':
        img2 = img2.astype(np.uint8)

    cv2.namedWindow("Image")
    cv2.imshow("Image", img2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def decode(a, img):
    a.read(20, 20, img)
    dis(a.decode_img)


class JSteg:
    def __init__(self):
        self.img = None
        self.dct_ = None
        self.dct_quantified = None
        self.max_info_len = 0
        self.row = 0
        self.col = 0
        self.available_info_len = 0
        self.encode_img = None
        self.decode_img = None
        self.test = None
        self.info_test = None

    def set_img(self, input_img):
        self.img = cv2.imread(input_img, flags=0).astype(np.float32)
        row, col = self.img.shape
        if not ((row / 8).is_integer() and (col / 8).is_integer()):
            print('像素横纵要为8的倍数')
            self.img = self.img[:int(row // 8) * 8, :int(col // 8) * 8]
            row, col = self.img.shape
        self.row = row
        self.col = col

    def write(self, pgm_file):
        # DCT & quantify
        array = self.img
        array = self.dct(array)
        self.dct_ = array
        array = self.quantify(array)
        self.dct_quantified = array

        self.info_test = array

        self.dct_quantified = self.dct_quantified.ravel()

        pgm = cv2.imread(pgm_file, 0)
        pgm_info = np.where(pgm > 127, 1, 0)  # *255

        info_len = pgm.shape[0] * pgm.shape[1]
        info_index = 0

        info = pgm_info.ravel()

        for i in range(self.col * self.row):
            if self._write(i, info[info_index]):
                # print('i=%d, info_index= %d, info[info_index]=%d' % (i, info_index, info[info_index]))
                #                print('self.dct_quantified[i]=%d'%self.dct_quantified[i])

                info_index += 1

            if info_index >= info_len:
                break

        self.test = self.dct_quantified.copy()

        print(self.test)

        temp = self.dct_quantified
        temp.resize(self.row, self.col)
        self.encode_img = temp

        temp = self.i_quantify(temp)
        temp = self.idct(temp)
        self.encode_img = temp

    def read(self, row, col, img_array):
        info_len = row * col
        # DCT & quantify
        array = img_array
        array = self.dct(array)
        self.dct_ = array
        array = self.quantify(array)
        self.dct_quantified = array

        info = []
        info_index = 0
        self.dct_quantified = self.dct_quantified.flatten()
        sequence_index = 0
        while True:
            if info_index >= info_len:
                break
            data = self._read(sequence_index)
            if data != None:
                info.append(data)
                info_index += 1
            sequence_index += 1

        info = np.array(info)
        info = info.astype(np.uint8)
        info.resize(row, col)
        info = info * 255

        self.decode_img = info

    # ====
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
        return True

    def _read(self, index):
        if self.dct_quantified[index] not in (-1, 1, 0):
            return self.dct_quantified[index] % 2
        else:
            return None

    def dct(self, full_array):
        row, col = full_array.shape
        full_array = full_array - 128
        for i in range(0, row, 8):
            for j in range(0, col, 8):
                # 也可以用np.lib.stride_tricks.as_strided(sudoku, shape=shape, strides=strides) 
                temp = full_array[i:i + 8, j:j + 8]
                full_array[i:i + 8, j:j + 8] = cv2.dct(temp)
        return full_array

    def idct(self, full_array):
        row, col = full_array.shape
        for i in range(0, row, 8):
            for j in range(0, col, 8):
                # 也可以用np.lib.stride_tricks.as_strided(sudoku, shape=shape, strides=strides) 
                temp = full_array[i:i + 8, j:j + 8]
                full_array[i:i + 8, j:j + 8] = cv2.idct(temp)
        full_array = full_array + 128
        return full_array

    def quantify(self, full_array):
        row, col = full_array.shape
        qu_array = np.array([[16, 11, 10, 16, 1, 1, 1, 1],
                             [12, 12, 14, 1, 1, 1, 1, 55],
                             [14, 13, 1, 1, 1, 1, 69, 56],
                             [14, 1, 1, 1, 1, 87, 80, 62],
                             [1, 1, 1, 1, 68, 109, 109, 77],
                             [1, 1, 1, 64, 81, 104, 104, 92],
                             [1, 1, 78, 87, 103, 121, 121, 101],
                             [1, 92, 95, 98, 112, 100, 100, 99]])
        for i in range(0, row, 8):
            for j in range(0, col, 8):
                temp = full_array[i:i + 8, j:j + 8]
                full_array[i:i + 8, j:j + 8] = np.around(temp / qu_array)
        self.dct_quantified = full_array

        self.available_info_len = row * col - (full_array == 0).sum() - (full_array == 1).sum()
        print('available_info_len = %d' % self.available_info_len)
        return full_array

    def i_quantify(self, full_array):
        row, col = full_array.shape
        qu_array = np.array([[16, 11, 10, 16, 1, 1, 1, 1],
                             [12, 12, 14, 1, 1, 1, 1, 55],
                             [14, 13, 1, 1, 1, 1, 69, 56],
                             [14, 1, 1, 1, 1, 87, 80, 62],
                             [1, 1, 1, 1, 68, 109, 109, 77],
                             [1, 1, 1, 64, 81, 104, 104, 92],
                             [1, 1, 78, 87, 103, 121, 121, 101],
                             [1, 92, 95, 98, 112, 100, 100, 99]])
        for i in range(0, row, 8):
            for j in range(0, col, 8):
                temp = full_array[i:i + 8, j:j + 8]
                full_array[i:i + 8, j:j + 8] = np.around(temp * qu_array)
        return full_array


if __name__ == '__main__':
    a = JSteg()
    a.set_img('7.pgm')
    a.write('0.pgm')
    dis(a.encode_img)
#    pgm = cv2.imread('0.pgm', flags=0).astype(np.float32)
#    pgm_info = np.where(pgm > 127, 1, 0)*255
#    a.read(20,20, a.encode_img)
#    dis(a.decode_img)
