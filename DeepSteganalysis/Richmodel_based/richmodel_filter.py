#! /usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2017/12/28 10:00
# @Author  : Shiyu Li
# @Software: PyCharm

import numpy as np

# RichModel Filter

# =========================================================
# 3*3 1st 2nd 3rd Edge Square(spam for all)
# num=8 1st
F33_1st_1 = np.array([[0, 0, 0],
                      [0, -1, 1],
                      [0, 0, 0]], dtype=np.float32)
F33_1st_2 = np.array([[0, 0, 1],
                      [0, -1, 0],
                      [0, 0, 0]], dtype=np.float32)
F33_1st_3 = np.array([[0, 1, 0],
                      [0, -1, 0],
                      [0, 0, 0]], dtype=np.float32)
F33_1st_4 = np.array([[1, 0, 0],
                      [0, -1, 0],
                      [0, 0, 0]], dtype=np.float32)
F33_1st_5 = np.array([[0, 0, 0],
                      [1, -1, 0],
                      [0, 0, 0]], dtype=np.float32)
F33_1st_6 = np.array([[0, 0, 0],
                      [0, -1, 0],
                      [1, 0, 0]], dtype=np.float32)
F33_1st_7 = np.array([[0, 0, 0],
                      [0, -1, 0],
                      [0, 1, 0]], dtype=np.float32)
F33_1st_8 = np.array([[0, 0, 0],
                      [0, -1, 0],
                      [0, 0, 1]], dtype=np.float32)
# num=4 2nd
F33_2nd_1 = np.array([[0, 0, 0],
                      [1, -2, 1],
                      [0, 0, 0]], dtype=np.float32)
F33_2nd_2 = np.array([[0, 0, 1],
                      [0, -2, 0],
                      [1, 0, 0]], dtype=np.float32)
F33_2nd_3 = np.array([[0, 1, 0],
                      [0, -2, 0],
                      [0, 1, 0]], dtype=np.float32)
F33_2nd_4 = np.array([[1, 0, 0],
                      [0, -2, 0],
                      [0, 0, 1]], dtype=np.float32)

# num=8 3rd
F33_3rd_1 = np.array([[1, 1, 1],
                      [0, -3, 0],
                      [0, 0, 0]], dtype=np.float32)
F33_3rd_2 = np.array([[0, 1, 1],
                      [0, -3, 1],
                      [0, 0, 0]], dtype=np.float32)
F33_3rd_3 = np.array([[0, 0, 1],
                      [0, -3, 1],
                      [0, 0, 1]], dtype=np.float32)
F33_3rd_4 = np.array([[0, 0, 0],
                      [0, -3, 1],
                      [0, 1, 1]], dtype=np.float32)
F33_3rd_5 = np.array([[0, 0, 0],
                      [0, -3, 0],
                      [1, 1, 1]], dtype=np.float32)
F33_3rd_6 = np.array([[0, 0, 0],
                      [1, -3, 0],
                      [1, 1, 0]], dtype=np.float32)
F33_3rd_7 = np.array([[1, 0, 0],
                      [1, -3, 0],
                      [1, 0, 0]], dtype=np.float32)
F33_3rd_8 = np.array([[1, 1, 0],
                      [1, -3, 0],
                      [0, 0, 0]], dtype=np.float32)
# num=4 Edge
F33_Edge_1 = np.array([[-1, 2, -1],
                       [2, -4, 2],
                       [0, 0, 0]], dtype=np.float32)
F33_Edge_2 = np.array([[0, 2, -1],
                       [0, -4, 2],
                       [0, 2, -1]], dtype=np.float32)
F33_Edge_3 = np.array([[0, 0, 0],
                       [2, -4, 2],
                       [-1, 2, -1]], dtype=np.float32)
F33_Edge_4 = np.array([[-1, 2, 0],
                       [2, -4, 0],
                       [-1, 2, 0]], dtype=np.float32)

# num=2 Square
F33_Square_1 = np.array([[-1, 2, -1],
                         [2, -4, 2],
                         [-1, 2, -1]], dtype=np.float32)
F33_Square_2 = np.array([[2, -1, 2],
                         [-1, -4, -1],
                         [2, -1, 2]], dtype=np.float32)

# =========================================================
# 5*5 Edge Square(spam for all)

# num=4 Edge
F55_Edge_1 = np.array([[-1, 2, -2, 2, -1],
                       [2, -6, 8, -6, 2],
                       [-2, 8, -12, 8, -2],
                       [0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0]], dtype=np.float32)
F55_Edge_2 = np.array([[-1, 2, -2, 0, 0],
                       [2, -6, 8, 0, 0],
                       [-2, 8, -12, 0, 0],
                       [2, -6, 8, 0, 0],
                       [-1, 2, -2, 0, 0]], dtype=np.float32)
F55_Edge_3 = np.array([[0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0],
                       [-2, 8, -12, 8, -2],
                       [2, -6, 8, -6, 2],
                       [-1, 2, -2, 2, -1]], dtype=np.float32)
F55_Edge_4 = np.array([[0, 0, -2, 2, -1],
                       [0, 0, 8, -6, 2],
                       [0, 0, -12, 8, -2],
                       [0, 0, 8, -6, 2],
                       [0, 0, -2, 2, -1]], dtype=np.float32)
# num=1 Square
F55_Square = np.array([[-1, 2, -2, 2, -1],
                       [2, -6, 8, -6, 2],
                       [-2, 8, -12, 8, -2],
                       [2, -6, 8, -6, 2],
                       [-1, 2, -2, 2, -1]], dtype=np.float32)

# #==============================================================
list_1st_33 = [F33_1st_1, F33_1st_2, F33_1st_3, F33_1st_4, F33_1st_5, F33_1st_6, F33_1st_7, F33_1st_8]
list_2nd_33 = list(map(lambda x: x / 2, [F33_2nd_1, F33_2nd_2, F33_2nd_3, F33_2nd_4]))
list_3rd_33 = list(map(lambda x: x / 3,
                       [F33_3rd_1, F33_3rd_2, F33_3rd_3, F33_3rd_4, F33_3rd_5, F33_3rd_6, F33_3rd_7, F33_3rd_8]))
list_Edge_33 = list(map(lambda x: x / 4, [F33_Edge_1, F33_Edge_2, F33_Edge_3, F33_Edge_4]))
list_Square_33 = list(map(lambda x: x / 4, [F33_Square_1, F33_Square_2]))

list_Edge_55 = list(map(lambda x: x / 12, [F55_Edge_1, F55_Edge_2, F55_Edge_3, F55_Edge_4]))
list_Square_55 = [F55_Square / 12]

list_33 = list_1st_33 + list_2nd_33 + list_3rd_33 + list_Edge_33 + list_Square_33
list_55 = list_Edge_55 + list_Square_55
