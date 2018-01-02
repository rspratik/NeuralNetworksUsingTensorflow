import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

tf.reset_default_graph()

def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))


def model(X, w_h1, w_o):
    h = tf.nn.sigmoid(tf.matmul(X, w_h1)) # this is a basic mlp
    return tf.matmul(h, w_o) # note that we dont take the softmax at the end because our cost fn does that for us

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
trX, trY, teX, teY = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels

size_h1 = tf.constant(625, dtype=tf.int32)

X = tf.placeholder("float", [None, 784])
Y = tf.placeholder("float", [None, 10])

w_h1 = init_weights([784, size_h1]) # create symbolic variables
w_o = init_weights([size_h1, 10])

py_x = model(X, w_h1, w_o)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=py_x, labels=Y)) # compute costs
train_op = tf.train.GradientDescentOptimizer(0.05).minimize(cost) # construct an optimizer
predict_op = tf.argmax(py_x, 1)

# Saving
saver = tf.train.Saver()
# Launch the graph in a session
with tf.Session() as sess:
    # you need to initialize all variables
    #tf.global_variables_initializer().run()
    saver.restore(sess, "./mlp/hidden_weights.ckpt")

    print("Picking up last saved state.See the accuracy trend by executing this program again and again.")
    for i in range(10):
        for start, end in zip(range(0, len(trX), 128), range(128, len(trX)+1, 128)):
            sess.run(train_op, feed_dict={X: trX[start:end], Y: trY[start:end]})
        print(i, np.mean(np.argmax(teY, axis=1) ==
                         sess.run(predict_op, feed_dict={X: teX})))
        # Save the variables to disk.
        save_path = saver.save(sess, "./mlp/hidden_weights.ckpt")
        print("Model saved in file: %s" % save_path)
