import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

cwd = "./mlp/hidden_weights"

def init_weights(shape, name):
    return tf.Variable(tf.random_normal(shape, stddev=0.01), name=name)

#Model with sigmoid activation function
def model_with_sig_activation(X, w_h1, w_o):
    with tf.name_scope("layer1_with_sigmoid_activation"):
        h = tf.nn.sigmoid(tf.matmul(X, w_h1)) # this is a basic mlp
        return tf.matmul(h, w_o) # note that we dont take the softmax at the end because our cost fn does that for us

#Model with relu activation function
def model_with_relu_activation(X, w_h1, w_o):
    with tf.name_scope("layer1_with_relu_activation"):
        h = tf.nn.relu(tf.matmul(X, w_h1)) # this is a basic mlp
        return tf.matmul(h, w_o) # note that we dont take the softmax at the end because our cost fn does that for us

#Model with leaky-relu activation function
def model_with_leakyrelu_activation(X, w_h1, w_o, alpha_leak):
    with tf.name_scope("layer1_with_leakyrelu_activation"):
        h = tf.nn.leaky_relu(tf.matmul(X, w_h1), alpha=alpha_leak) # this is a basic mlp
        return tf.matmul(h, w_o) # note that we dont take the softmax at the end because our cost fn does that for us


mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
trX, trY, teX, teY = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels

size_h1 = tf.constant(625, dtype=tf.int32)

X = tf.placeholder("float", [None, 784], name="X")
Y = tf.placeholder("float", [None, 10], name="Y")

w_h1 = init_weights([784, size_h1], "w_h") # create symbolic variables
w_o = init_weights([size_h1, 10], "w_o")

#Saving weights
tf.summary.histogram("w_h1", w_h1)
tf.summary.histogram("w_o1", w_o)

#tf.summary.scalar("w_h", w_h1)
#tf.summary.scalar("w_o", w_o)

'''
 We can change the model according to the desired activation function and compare the performance by loading
 the root directory in tensor-board where multiple log folder corresponds to different models(each activation fn.) 
'''
#py_x = model_with_sig_activation(X, w_h1, w_o)
#py_x = model_with_relu_activation(X, w_h1, w_o)
#Compared in between alpha=0.1,0.2,0.3 and found 0.1 to be more effective.
py_x = model_with_leakyrelu_activation(X, w_h1, w_o, 0.1)

with tf.name_scope("cost"):
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=py_x, labels=Y)) # compute costs
    train_op = tf.train.GradientDescentOptimizer(0.05).minimize(cost) # construct an optimizer
    tf.summary.scalar("cost", cost)

with tf.name_scope("accuracy"):
    correct_pred = tf.equal(tf.argmax(Y, 1), tf.argmax(py_x, 1)) # Count correct predictions
    acc_op = tf.reduce_mean(tf.cast(correct_pred, "float")) # Cast boolean to float to average
    # Add scalar summary for accuracy tensor
    tf.summary.scalar("accuracy", acc_op)

# Saving
saver = tf.train.Saver()

with tf.Session() as sess:
    # tensorboard --logdir=./logs/mlp-2a
    # Use different dir for writing diff models so that comparison go easy.
    with tf.summary.FileWriter(cwd, sess.graph) as writer:
        #writer = tf.summary.FileWriter(cwd, sess.graph)
        merged = tf.summary.merge_all()

        tf.global_variables_initializer().run()

        for i in range(5):
            for start, end in zip(range(0, len(trX), 128), range(128, len(trX)+1, 128)):
                sess.run(train_op, feed_dict={X: trX[start:end], Y: trY[start:end]})
            summary, acc = sess.run([merged, acc_op], feed_dict={X: teX, Y: teY})
            writer.add_summary(summary, i)  # Write summary
            print(i, acc)                   # Report the accuracy

    # Save the variables to disk.
    save_path = saver.save(sess, "./mlp/hidden_weights.ckpt")
    print("Model saved in file: %s" % save_path)