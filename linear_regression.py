#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import xlrd

import itertools
import numpy as np
import tensorflow as tf

# argument parse
parser = argparse.ArgumentParser(prog='linear regression')
parser.add_argument("-i", "--inputs", type=str, help="input data")
args = parser.parse_args()

# parse data
data = xlrd.open_workbook(args.inputs, encoding_override='ascii')
table = data.sheets()[0]

features = np.array(table.col_values(0)[1:], dtype=np.float32)
labels = np.array(table.col_values(1)[1:], dtype=np.float32)

features_placeholder = tf.placeholder(tf.float32, features.shape)
labels_placeholder = tf.placeholder(tf.float32, labels.shape)

# parameters
batch_size = 8
num_epochs = 100
shuffle_size = 20
learning_rate = 0.01

dataset = tf.data.Dataset \
        .from_tensor_slices((features_placeholder, labels_placeholder)) \
        .batch(batch_size) \
        .shuffle(buffer_size=shuffle_size) \
        .repeat(num_epochs)

iterator = dataset.make_initializable_iterator()

x = tf.placeholder(dtype=tf.float32, shape=[None])
y = tf.placeholder(dtype=tf.float32, shape=[None])

#W = tf.Variable(tf.random_normal([]), dtype=tf.float32)
#b = tf.Variable(tf.random_normal([]), dtype=tf.float32)
W = tf.Variable(0., tf.float32)
b = tf.Variable(0., tf.float32)
linear_model = W * x + b

def huber_loss(labels, predictions, delta=1.0):
    residual = tf.abs(predictions - labels)
    condition = tf.less(residual, delta)
    small_res = tf.square(residual) * 0.5
    large_res = delta * residual - tf.square(delta) * 0.5
    return tf.where(condition, small_res, large_res)

#loss = tf.reduce_sum(tf.square(y - linear_model))
loss = huber_loss(y, linear_model)
optimizer = tf.train.GradientDescentOptimizer(learning_rate)
train = optimizer.minimize(loss)

with tf.Session() as ss:
    init = tf.global_variables_initializer()
    ss.run(init)
    data_dict = {features_placeholder: features, labels_placeholder: labels}
    ss.run(iterator.initializer, feed_dict=data_dict)

    next_batch = iterator.get_next()

    for i in itertools.count():
        try:
            _x, _y = ss.run(next_batch)
            feed_dict = dict(zip((x, y), [_x, _y]))
            ss.run(train, feed_dict)
            if i % 50 == 0:
                res = ss.run([loss, W, b], feed_dict=feed_dict)
                print('loss: [{}], W: [{}], b: [{}]'.format(*res))
        except tf.errors.OutOfRangeError:
            break

