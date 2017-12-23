#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-

import numpy as np
import sklearn.datasets
import tensorflow as tf


def input_fn(features, labels, shuffle=False, epochs=1):
    dataset = tf.data.Dataset.from_tensor_slices((features, labels))
    if shuffle:
        dataset = dataset.shuffle(buffer_size=64)

    dataset = dataset.repeat(epochs)
    dataset = dataset.batch(2)
    iterator = dataset.make_one_shot_iterator()
    next_batch = iterator.get_next()
    return next_batch
    

def model_fn(features, labels, mode, params):
    #batch_size = None
    W = tf.get_variable('W', [4, 3], dtype=tf.float64)
    b = tf.get_variable('b', [3], dtype=tf.float64)
    y = tf.add(tf.matmul(features, W), b)
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=y))
    global_step = tf.train.get_global_step()
    optimizer = tf.train.GradientDescentOptimizer(params.get('learning_rate', 0.01))
    train = tf.group(optimizer.minimize(cross_entropy),
            tf.assign_add(global_step, 1))

    return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=y,
            loss=cross_entropy,
            train_op=train)


def main():
    def separate_train_test(train_data_len):
        iris = sklearn.datasets.load_iris()
        index = list(range(len(iris.data)))
        def onehot(data):
            v = [0, 0, 0]
            v[data] = 1
            return v
        np.random.shuffle(index)
        features, labels = iris.data, iris.target
        train_features = np.array([features[i] for i in index[:train_data_len]])
        train_labels = np.array([onehot(labels[i]) for i in index[:train_data_len]], dtype=np.float64)
        train_data = (train_features, train_labels)

        test_features = np.array([features[i] for i in index[train_data_len:]])
        test_labels = np.array([onehot(labels[i]) for i in index[train_data_len:]], dtype=np.float64)
        test_data = (test_features, test_labels)
        return (train_data, test_data)

    train_data, test_data = separate_train_test(120)

    model_params = {'learning_rate': 0.01}
    estimator = tf.estimator.Estimator(model_fn=model_fn, params=model_params)
    estimator.train(input_fn=lambda: input_fn(*train_data, shuffle=True, epochs=100), steps=100)
    train_metrics = estimator.evaluate(input_fn=lambda: input_fn(*train_data))
    test_metrics = estimator.evaluate(input_fn=lambda: input_fn(*test_data))
    print("train metrics: %r" % train_metrics)
    print("eval metrics: %r" % test_metrics)


if __name__ == "__main__":
    main()
