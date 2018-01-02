'''
1.Added a 2nd hidden layer.
2.Shortened the number of epochs to 30-50.
3.Changed the number of neurons in each layer.
4.Changed the batch size from 128 to 64.
'''

import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data


def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))


def model(X, w_h1, w_h2, w_o):
    # X is [N x 784] ; w_h1 is [784 x 498]; so h1 would be [N x 498]
    h1 = tf.nn.sigmoid(tf.matmul(X, w_h1)) # this is a basic mlp

    # h1 is [N x 498] ; w_h2 is [498 x 712]; so h2 would be [N x 712]
    h2 = tf.nn.sigmoid(tf.matmul(h1, w_h2))  # this is a basic mlp

    #h2 would be [N x 712]; w_o is [712 x 10]; so (h2 * w_o) will be [N x 10]
    return tf.matmul(h2, w_o) # note that we don't take the softmax at the end because our cost fn does that for us

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
trX, trY, teX, teY = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels

size_h1 = tf.constant(498, dtype=tf.int32)
# let size_h2 = 712
size_h2 = tf.constant(712, dtype=tf.int32)

X = tf.placeholder("float", [None, 784])
Y = tf.placeholder("float", [None, 10])

w_h1 = init_weights([784, size_h1]) # create symbolic variables
w_h2 = init_weights([size_h1, size_h2]) # create symbolic variables
w_o = init_weights([size_h2, 10])

py_x = model(X, w_h1, w_h2, w_o)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=py_x, labels=Y)) # compute costs
train_op = tf.train.GradientDescentOptimizer(0.05).minimize(cost) # construct an optimizer
predict_op = tf.argmax(py_x, 1)

# Launch the graph in a session
with tf.Session() as sess:
    # you need to initialize all variables
    tf.global_variables_initializer().run()
    # printing the range with the batch size =64
    print(range(0,len(trX),64))
    # reducing the epoch to 30
    for i in range(30):
        # reducing the batch size to 64
        for start, end in zip(range(0, len(trX), 64), range(64, len(trX)+1, 64)):
            #print(start," --- " ,end)
            sess.run(train_op, feed_dict={X: trX[start:end], Y: trY[start:end]})
        print(i, np.mean(np.argmax(teY, axis=1) ==
                         sess.run(predict_op, feed_dict={X: teX})))
