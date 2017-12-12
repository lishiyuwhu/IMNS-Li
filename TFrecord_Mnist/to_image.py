#! /usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2017/12/10 14:10
# @Author  : Shiyu Li
# @Software: PyCharm

# !/usr/bin/env python
# -*- coding: utf-8 -*-


from PIL import Image
import struct, os

root = os.getcwd()
for i in range(10):
    os.makedirs(root + '/pngtrain/' + str(i))

filename1 = 'train-images.idx3-ubyte'
filename2 = 'train-labels.idx1-ubyte'
#
# filename1 = 't10k-images.idx3-ubyte'
# filename2 = 't10k-labels.idx1-ubyte'

# label

f2 = open(filename2, 'rb')
index2 = 0
buf2 = f2.read()
f2.close()
magic2, labels2 = struct.unpack_from('>II', buf2, index2)
index2 += struct.calcsize('>II')
labelArr = [0] * labels2
for x in range(labels2):
    labelArr[x] = int(struct.unpack_from('>B', buf2, index2)[0])
    index2 += struct.calcsize('>B')

# image
f = open(filename1, 'rb')
index = 0
buf = f.read()
f.close()
magic, images, rows, columns = struct.unpack_from('>IIII', buf, index)
index += struct.calcsize('>IIII')

for i in range(images):
    image = Image.new('L', (columns, rows))

    for x in range(rows):
        for y in range(columns):
            image.putpixel((y, x), int(struct.unpack_from('>B', buf, index)[0]))
            index += struct.calcsize('>B')

    print('save ' + str(i) + 'image')
    image.save('pngtrain/' + str(labelArr[i]) + '/' + str(i) + '.png')
