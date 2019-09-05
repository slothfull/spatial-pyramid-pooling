#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
@time:2018/11/12 下午7:57
@author:bigmelon
"""
import tensorflow as tf
from morvan_tutorials.divide_spatial_pyramid_pooling_test import dongh_pyramid_pooling

from tensorflow.examples.tutorials.mnist import input_data
# number 1 to 10 data

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
batch_size = 100


def compute_accuracy(v_xs, v_ys):
    global prediction
    y_pre = sess.run(prediction, feed_dict={xs: v_xs, keep_prob: 1})
    correct_prediction = tf.equal(tf.argmax(y_pre, 1), tf.argmax(v_ys, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    # 下面的accuracy没有显示......
    tf.summary.scalar("accuracy", accuracy)
    result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys})
    return result


def weight_variable(shape):
    # using truncated normal variables to initialize weight
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, w):
    # stride [1, x_movement, y_movement, 1]
    # padding = 'SAME' do not change the size after conv
    return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_1x2(x):
    # for ksize=strides=[1,2,2,1]
    # we can see the max-pooling is implemented only on height/width but not on batch or channels
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def max_pool_2x2(x):
    # dongh_pyramid_pooling(input_tensor, height_divide, height_divide, pooling='max', shuffle=True)
    # return (batch_size, height_divide, width_divide, channels)
    return dongh_pyramid_pooling(x, height_divide=2, width_divide=2, pooling='max', shuffle=True)


# define placeholder for inputs to network
with tf.name_scope('placeholder'):
    with tf.name_scope('placeholder_xs'):
        xs = tf.placeholder(tf.float32, [None, 784])
    with tf.name_scope('placeholder_ys'):
        ys = tf.placeholder(tf.float32, [None, 10])
        # define placeholder for dropout to reduce over fitting
    with tf.name_scope('placeholder_dropout'):
        keep_prob = tf.placeholder(tf.float32)
        # [-1,,,]??????????????
        #  [-1, 28, 28, 1]) = [ , , , channel] for gray image channel = 1   for rgb channel = 3
        #  784 = 28 X 28 for one "mnist picture"
        x_image = tf.reshape(xs, [-1, 28, 28, 1])


# conv1 layer #
# [5, 5, 1, 32] : patch = 5X5 insize = 1 outsize(feature map) = 32
with tf.name_scope('layer1'):
    with tf.name_scope('w_conv1'):
        # filter_shape=5x5 in=1 out=32
        W_conv1 = weight_variable([5, 5, 1, 32])
    with tf.name_scope('b_conv1'):
        b_conv1 = bias_variable([32])
    with tf.name_scope('h_conv1'):
        # x_image=(100, 28, 28, 1) w_conv1=(5, 5, 1, 32) -> h_conv1=(100, 28, 28, 32)
        h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
        # h_conv1=(100, 28, 28, 32) -> h_pool1=(100, 14, 14, 32)
        h_pool1 = max_pool_1x2(h_conv1)


# conv2 layer #
# [5, 5, 32, 64] : patch = 5X5 insize =32 outsize =64
with tf.name_scope('layer2'):
    with tf.name_scope('w_conv2'):
        W_conv2 = weight_variable([5, 5, 32, 64])
    with tf.name_scope('b_conv2'):
        b_conv2 = bias_variable([64])
    with tf.name_scope('h_conv2'):
        # h_conv2=(100, 14, 14, 64)
        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
        # h_conv2=(100, 14, 14, 64) -> h_pool2 = (100, 1, 4, 64) = 25600
        h_pool2 = max_pool_2x2(h_conv2)


# func1 layer #
with tf.name_scope('w_fc1'):
    W_fc1 = weight_variable([2*2*64, 64])
with tf.name_scope('wb_fc1'):
    b_fc1 = bias_variable([64])
    # [n_samples, 7, 7, 64] ->> [n_samples, 7 * 7 * 64]
    h_pool2_flat = tf.reshape(h_pool2, [-1, 2*2*64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)


# func2 layer prediction layer #
with tf.name_scope('w_fc2'):
    W_fc2 = weight_variable([64, 10])
with tf.name_scope('wb_fc2'):
    b_fc2 = bias_variable([10])
    prediction = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
    # h_fc2_drop = tf.nn.dropout(b_fc2, keep_prob)

# the error between prediction and real data
with tf.name_scope('cross_entropy'):
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction), reduction_indices=[1], name='reduce_sum'),
                                   name='reduce_mean')
    tf.summary.scalar('cross_entropy', cross_entropy)

with tf.name_scope('Train'):
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)


# important step
# tf.initialize_all_variables() no long valid form
# 2017-03-02 if using tensorflow >= 0.12
# init = tf.initialize_all_variables()
init = tf.global_variables_initializer()

sess = tf.Session()
#  write out a "logs" file
merged = tf.summary.merge_all()  # this row can be deleted with no effect
train_writer = tf.summary.FileWriter("logs/", sess.graph)
sess.run(init)


for i in range(200):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={xs: batch_xs, ys: batch_ys, keep_prob: 0.5})
    if i % 10 == 0:
        print(compute_accuracy(mnist.test.images[:1000], mnist.test.labels[:1000]))
        train_result = sess.run(merged, feed_dict={xs: batch_xs, ys: batch_ys, keep_prob: 1})
        train_writer.add_summary(train_result, i)
#  steps to get the tensorboard graphs \ histograms \ scalars = pure quantity
#  attention that the "logs/" path is :
#  /Users/mac/Desktop/untitled/logs/events.out.tfevents.1511424334.twistfatezzz.local
#  open terminal:
#  cd /Users/mac/Desktop/untitled/morvan_tutorials/logs/train
#  key into the terminal: tensorboard --logdir="logs/"
#  go to the web(google chrome is recommended): http://localhost:6006
