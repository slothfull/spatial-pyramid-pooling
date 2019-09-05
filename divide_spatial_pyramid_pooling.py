#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
@time:2018/11/12 下午7:31
@author:bigmelon

divide_spatial_pyramid_pooling-ops:
->implement the op that divide arbitrary size feature map(axb) into arbitrary sub-feature-maps
->then do max/min/avg pooling to each sub-feature-map to derive mxn dimension tensor output for arbitrary feature input
->this op can solve the arbitrary input of networks...

->the drawbacks of dongh_pooling_layer:
cannot receive dynamic input batch_size for the tf.strided_slice() has to use the concrete end()=batch_size  but not none_type

->how to fix the problem above???????
"""
import tensorflow as tf
import numpy as np
import math
import random


# a = 3  # height of the feature map
# b = 4  # width of the feature map
#
# m = 2  # the feature map is divided by 2 in height
# n = 3  # the feature map is divided by 3 in width
#
# pool = 'max'
# # shuffle = True
# batch_size = 2
# channels = 2
#
#
# # set the feature map as type(batch_size, height, width, channels) = NHWC
# feature_map = np.arange(48)
# feature_map = np.reshape(feature_map, (2, 3, 4, 2))
# print(feature_map)
#
# # define the placeholder to hold the feature map tensor
# feature_map_tensor = tf.placeholder(tf.float32, [batch_size, a, b, channels])


def derive_divide_list(para_a, para_m):
    """derive the split proportion list...
    :param para_a: width or height of the feature map
    :param para_m: the num of splits to the width or height
    :return: the split proportion list...
    """
    x = []  # hold the width of each spatial_bin
    while para_m != 0:
        xx = math.ceil(para_a / para_m)
        x.append(xx)
        para_a = para_a - xx
        para_m = para_m - 1
    return x


def derive_sub_feature_map_list(para_feature_map_tensor, para_d0, para_d1):
    """derive sub feature map lists...
    :param para_feature_map_tensor: feature map(a tensor) for split...
    :param para_d0: split proportion list for height of feature map
    :param para_d1: split proportion list for width of feature map
    :return: sub feature map lists...
    """
    # split firstly from dimension 1 --->height
    # s0, s1 = tf.split(value=x, num_or_size_splits=d0=[1, 2], axis=1)
    split0 = tf.split(value=para_feature_map_tensor, num_or_size_splits=para_d0, axis=1, name='split_in_height')

    # split secondly from dimension 2 --->width
    para_sub_feature_maps_dict = []
    for _ in range(len(split0)):
        # d1=[2, 1, 1]
        split1 = tf.split(value=split0[_], num_or_size_splits=para_d1, axis=2, name='split_in_width')
        # type(sub_feature_maps_dict)=list np.array(sub_feature_maps_dict).shape=(2, 3)
        para_sub_feature_maps_dict.append(list(split1))
        # xxx:
        # 对x0进行分块得到:
        # [[<tf.Tensor 'split_7:0' shape=(2, 1, 2, 2) dtype=int64>,  --> x00  x(height,width)
        #   <tf.Tensor 'split_7:1' shape=(2, 1, 1, 2) dtype=int64>,  --> x01
        #   <tf.Tensor 'split_7:2' shape=(2, 1, 1, 2) dtype=int64>], --> x02
        # 对x1进行分块得到:
        #  [<tf.Tensor 'split_8:0' shape=(2, 2, 2, 2) dtype=int64>,  --> x10
        #   <tf.Tensor 'split_8:1' shape=(2, 2, 1, 2) dtype=int64>,  --> x11
        #   <tf.Tensor 'split_8:2' shape=(2, 2, 1, 2) dtype=int64>]] --> x12
    return para_sub_feature_maps_dict


def dongh_pyramid_pooling(input_tensor, height_divide, width_divide, pooling='max', shuffle=True):
    """divide-pyramid-pooling by dongh
    :param input_tensor: with shape as (para_batch_size, height, width, channels)
    :param height_divide: integer to divide the height of input tensor
    :param width_divide: integer to divide the width of input tensor
    :param pooling: 'max'->tf.nn.max_pooling | 'avg'->tf.nn.avg_pooling
    :param shuffle: True ->shuffle the sub-divided-maps  | False ->not shuffle the sub-divided-maps
    :return: a tensor with shape:(para_batch_size, 1, height_divide*width_divide, channels)
    """
    # para_batch_size_for_slice = input_tensor.get_shape().as_list()[0]
    # para_batch_size_for_slice=None | type(.)=Nonetype is not compatible for tf.strided_slice(int32)
    height_for_divide = input_tensor.get_shape().as_list()[1]
    width_for_divide = input_tensor.get_shape().as_list()[2]
    input_channels = input_tensor.get_shape().as_list()[3]

    # only using tf.shape can return a dimension of tensor to be used in tf.zeros() but for concat(tf.zeros(), tf)
    para_batch_size = tf.shape(input_tensor)[0]
    # height = tf.shape(input_tensor)[1]
    # width = tf.shape(input_tensor)[2]
    # input_channels = tf.shape(input_tensor)[3]

    d0 = derive_divide_list(height_for_divide, height_divide)  # d0 = [2, 1]
    d1 = derive_divide_list(width_for_divide, width_divide)  # d1 = [2, 1, 1]

    # (if not shuffled: the size of sub feature maps always from bigger to smaller)
    # random split list d0/d1 to avoid unbalanced split to undermine the behavior of pooling layer...
    if shuffle:
        random.shuffle(d0)  # d0 = [1, 2]
        random.shuffle(d1)  # d1 = [2, 1, 1]
    else:
        pass

    sub_feature_maps_dict = derive_sub_feature_map_list(input_tensor, d0, d1)

    # initialize the result_tensor  dtype=tf.float32
    result_tensor = tf.zeros([para_batch_size, 1, 1, input_channels])
    # result_tensor = tf.zeros([batch_size, 1, 1, input_channels])

    for i in range(np.array(sub_feature_maps_dict).shape[0]):  # i=height
        for j in range(np.array(sub_feature_maps_dict).shape[1]):  # j=width
            pool_height = sub_feature_maps_dict[i][j].get_shape().as_list()[1]  # assign the pool_height=sub_feature_map_height
            pool_width = sub_feature_maps_dict[i][j].get_shape().as_list()[2]  # assign the pool_width=sub_feature_map_width
            # only pooling with height/width of sub feature maps...
            # for the feature map has the same size of kernels and stride.shape = the kernel.shape
            # so 2 kinds of paddings are the same...
            # sub_feature_map_pool_output.shape=(para_batch_size, 1, 1, channels)
            if pooling == 'max':
                sub_feature_map_pool_output = tf.nn.max_pool(sub_feature_maps_dict[i][j],
                                                             ksize=[1, pool_height, pool_width, 1],
                                                             strides=[1, pool_height, pool_width, 1],
                                                             padding='VALID', data_format="NHWC", name=None)
            elif pooling == 'avg':
                sub_feature_map_pool_output = tf.nn.avg_pool(sub_feature_maps_dict[i][j],
                                                             ksize=[1, pool_height, pool_width, 1],
                                                             strides=[1, pool_height, pool_width, 1],
                                                             padding='VALID', data_format="NHWC", name=None)
            else:
                sub_feature_map_pool_output = None
                exit("wrong type of pooling!")
            # use tf.concat() to get the sub_pool output together...
            result_tensor = tf.concat([result_tensor, sub_feature_map_pool_output], axis=2, name="concat")
    # result_tensor.shape=(para_batch_size, 1, mxn+1, channels)
    # e.g. for a feature map divided into 16 sub-feature-maps: the 'output-tensor-length' should be 4x4+1
    # '+1' means the first initial additional tensor for use of tf.concat() and it should be deleted by 'tf.strided_slice'
    """
    此处的tf.strided_slice限制了batch_size不能是none_type 网络如果含有dongh_spatial_pyramid_pooling层 就只能够接受batch_size=100
    的数据 对于训练 是没有影响的 但是对于测试 希望拿测试集合的1000个数据作为一个batch送进网络进行测试就不行 
    所以此处的batch_size限制了 tf.placeholder()所接受的none_type
    
    """
    # pyramid_sub_feature_output = tf.strided_slice(result_tensor, begin=tf.convert_to_tensor([0, 0, 1, 0], dtype=tf.int32),
    #                                               end=tf.convert_to_tensor([batch_size, 1, height_divide*width_divide+1, input_channels], dtype=tf.int32),
    #                                               strides=tf.convert_to_tensor([1, 1, 1, 1], dtype=tf.int32))
    # pyramid_sub_feature_output = tf.strided_slice(result_tensor, begin=tf.convert_to_tensor([0, 0, 1, 0], dtype=tf.int32),
    #                                               end=tf.convert_to_tensor([-1, 1, height_divide*width_divide+1, input_channels], dtype=tf.int32),
    #                                               strides=tf.convert_to_tensor([1, 1, 1, 1], dtype=tf.int32))
    # pyramid_sub_feature_output = tf.strided_slice(result_tensor, begin=[0, 0, 1, 0], end=[-1, 1, height_divide*width_divide+1, input_channels],
    #                                               strides=[1, 1, 1, 1])
    """
    感谢大神的回答! from:https://stackoverflow.com/questions/39054414/tensorflow-using-tf-slice-to-split-the-input
    使用tf.slice的时候 由于不用传递结尾index(end) 所以可以直接传递从begin开始的slice的长度(size)即可 此时就可以通过传递-1来实现编译器
    自动找到?对应的结尾... hahahahahahahah!
    """
    pyramid_sub_feature_output = tf.slice(input_=result_tensor, begin=[0, 0, 1, 0], size=[-1, 1, height_divide * width_divide, input_channels])
    return pyramid_sub_feature_output


# # return an op to initialize all global variables...
# init = tf.global_variables_initializer()
#
# with tf.Session() as sess:
#     # initialize the tf variables through the op...
#     sess.run(init)
#
#     # final_pyramid_pooling_result.shape=(2, 1, 6, 2)
#     final_pyramid_pooling_result = sess.run(dongh_pyramid_pooling(feature_map_tensor, m, n, pooling='max', shuffle=True),
#                                             feed_dict={feature_map_tensor: feature_map})
