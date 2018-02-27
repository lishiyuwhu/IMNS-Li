#! /usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2018/2/27 15:15
# @Author  : Shiyu Li
# @Software: PyCharm

import numpy as np
import tensorlayer as tl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

X_train, y_train, _, _, X_test, y_test = \
    tl.files.load_mnist_dataset(shape=(-1, 784))

X_train = np.asarray(X_train, dtype=np.float32)
y_train = np.asarray(y_train, dtype=np.int32)
X_test = np.asarray(X_test, dtype=np.float32)
y_test = np.asarray(y_test, dtype=np.int32)


def Review_img():
    image_list = X_train[:9]
    image_list_labels = y_train[:9]
    fig = plt.figure(1, (5., 5.))
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                     nrows_ncols=(3, 3),  # creates 2x2 grid of axes
                     axes_pad=0.3,  # pad between axes in inch.
                     )

    for i in range(len(image_list)):
        image = image_list[i].reshape(28, 28)
        grid[i].imshow(image)
        grid[i].set_title('Label: {0}'.format(image_list_labels[i].argmax()))

    plt.show()


if __name__ =='__main__':
    print(y_test[0])
