import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

batch_size = 128
test_size = 256

'''
With 2 convolution layer.
Notes:
1.Please use different logdirs for comparison of results.
2.Graph,Plots can be seen using tensorboard by pointing to following cwd logdir
'''
cwd = "./cnn/logs"

def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))


def model(X, w1, w2, w_fc, w_o, p_keep_conv, p_keep_hidden, act):
    # Y  [None, 10])

    '''Convulution layer1'''
    # l1a shape=(?, 28, 28, 32) ; X[None, 28, 28, 1] ;  W1[3, 3, 1, 32]
    l1a = act(tf.nn.conv2d(X, w1, strides=[1, 1, 1, 1], padding='SAME'))
    # l1 shape=(?, 14, 14, 32) ; Downsampling l1a by factor of 2 .
    l1 = tf.nn.max_pool(l1a, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    l1 = tf.nn.dropout(l1, p_keep_conv)

    '''Convulution layer2 with doubled feature maps(64)'''
    # l2a shape=(?, 14, 14, 64) ; l1[None,14,14,32] ;  W2[3, 3, 32, 64]
    l2a = tf.nn.relu(tf.nn.conv2d(l1, w2, strides=[1, 1, 1, 1], padding='SAME'))
    # l2 shape=(?, 14, 14, 64)
    l2 = tf.nn.max_pool(l2a, ksize=[1, 1, 1, 1], strides=[1, 1, 1, 1], padding='SAME')
    l2 = tf.nn.dropout(l2, p_keep_conv)

    '''Dense layer'''
    # reshape l2[None,14,14,64] to l3[None, 14*14*64]  : reducing dimensions.
    l3 = tf.reshape(l2, [-1, w_fc.get_shape().as_list()[0]]) #l3[None, 14*14*64]
    l3 = tf.nn.dropout(l3, p_keep_conv)

    '''Output layer'''
    l4 = act(tf.matmul(l3, w_fc)) #l3[None, 14*14*64]  and  W_FC[64*14*14, 625]
    l4 = tf.nn.dropout(l4, p_keep_hidden) #l4[none, 625]

    pyx = tf.matmul(l4, w_o)  #l4[none, 625] w_o [625, 10]
    return pyx # [none,10]

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
trX, trY, teX, teY = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels
trX = trX.reshape(-1, 28, 28, 1)  # 28x28x1 input img
teX = teX.reshape(-1, 28, 28, 1)  # 28x28x1 input img

X = tf.placeholder("float", [None, 28, 28, 1])
Y = tf.placeholder("float", [None, 10])

# w = init_weights([3, 3, 1, 32])       # 3x3x1 conv, 32 outputs
# w_fc = init_weights([32 * 14 * 14, 625]) # FC 32 * 14 * 14 inputs, 625 outputs
# w_o = init_weights([625, 10])         # FC 625 inputs, 10 outputs (labels)

w1 = init_weights([3, 3, 1, 32])       # 3x3x1 conv, 32 outputs
w2 = init_weights([3, 3, 32, 64])       # 3x3x32 conv, 64 outputs

w_fc = init_weights([64 * 14 * 14, 625]) # FC 64 * 14 * 14 inputs, 625 outputs
w_o = init_weights([625, 10])         # FC 625 inputs, 10 outputs (labels)

p_keep_conv = tf.placeholder("float")
p_keep_hidden = tf.placeholder("float")
# change the act argument to play with diff activation funs(tf.nn.leaky_relu, tf.nn.relu, etc) and compare their accuracy in tensorboard.
py_x = model(X, w1, w2, w_fc, w_o, p_keep_conv, p_keep_hidden, act=tf.nn.leaky_relu)


#cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=py_x, labels=Y))
#train_op = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(cost)
#predict_op = tf.argmax(py_x, 1)
'''
Creating separate blocks to output everything in tensorboard systematically.
'''

with tf.name_scope("cost"):
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=py_x, labels=Y))
    train_op = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(cost)
    tf.summary.scalar("cost", cost)


with tf.name_scope("accuracy"):
    correct_pred = tf.equal(tf.argmax(Y, 1), tf.argmax(py_x, 1)) # Count correct predictions
    acc_op = tf.reduce_mean(tf.cast(correct_pred, "float")) # Cast boolean to float to average
    # Add scalar summary for accuracy tensor
    tf.summary.scalar("accuracy", acc_op)


# Launch the graph in a session
with tf.Session() as sess:
    # you need to initialize all variables
    with tf.summary.FileWriter(cwd, sess.graph) as writer:
        merged = tf.summary.merge_all()
        tf.global_variables_initializer().run()

        for i in range(5):
            training_batch = zip(range(0, len(trX), batch_size),
                                 range(batch_size, len(trX)+1, batch_size))
            for start, end in training_batch:
                #Enable/Disable the dropout values from  p_keep_conv: 0.8, p_keep_hidden: 0.5 to  p_keep_conv: 1, p_keep_hidden: 1}
                sess.run(train_op, feed_dict={X: trX[start:end], Y: trY[start:end], p_keep_conv: 1, p_keep_hidden: 1})

            test_indices = np.arange(len(teX)) # Get A Test Batch
            np.random.shuffle(test_indices)
            test_indices = test_indices[0:test_size]

            #print(i, np.mean(np.argmax(teY[test_indices], axis=1) == sess.run(predict_op, feed_dict={X: teX[test_indices], p_keep_conv: 1.0, p_keep_hidden: 1.0})))
            summary, acc = sess.run([merged, acc_op], feed_dict={X: teX[test_indices],Y: teY[test_indices], p_keep_conv: 1.0, p_keep_hidden: 1.0})
            writer.add_summary(summary, i)  # Write summary
            print(i, acc)
