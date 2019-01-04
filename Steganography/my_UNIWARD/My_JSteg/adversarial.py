#! /usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2018/2/27 14:41
# @Author  : Shiyu Li
# @Software: PyCharm


# if __name__ == '__main__':
#     main()

import tensorflow as tf
import tensorlayer as tl


# 初始化
image = tf.Variable(tf.zeros((299, 299, 3)))
x = tf.placeholder(tf.float32, (299, 299, 3))

x_hat = image # our trainable adversarial input
assign_op = tf.assign(x_hat, x)


# 梯度下降

learning_rate = tf.placeholder(tf.float32, ())
y_hat = tf.placeholder(tf.int32, ())

labels = tf.one_hot(y_hat, 1000)
loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=[labels])
optim_step = tf.train.GradientDescentOptimizer(
    learning_rate).minimize(loss, var_list=[x_hat])


# 投影步骤

epsilon = tf.placeholder(tf.float32, ())

below = x - epsilon
above = x + epsilon
projected = tf.clip_by_value(tf.clip_by_value(x_hat, below, above), 0, 1)
with tf.control_dependencies([projected]):
    project_step = tf.assign(x_hat, projected)


# RUN

demo_epsilon = 2.0 / 255.0  # a really small perturbation
demo_lr = 1e-1
demo_steps = 100
demo_target = 924  # "guacamole"

# initialization step
sess.run(assign_op, feed_dict={x: img})

# projected gradient descent
for i in range(demo_steps):
    # gradient descent step
    _, loss_value = sess.run(
        [optim_step, loss],
        feed_dict={learning_rate: demo_lr, y_hat: demo_target})
    # project step
    sess.run(project_step, feed_dict={x: img, epsilon: demo_epsilon})
    if (i + 1) % 10 == 0:
        print('step %d, loss=%g' % (i + 1, loss_value))

adv = x_hat.eval()  # retrieve the adversarial example
