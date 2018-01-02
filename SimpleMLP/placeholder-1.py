'''
1. Added a variable, c, and added it to a and b. Confirmed the correct evaluation of your modified model.
2. Added a constant d that is used to multiply the addition of a, b and c. Confirmed the correct evaluation of your model.
'''

import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)

#Added a variable c
c = tf.Variable([[1,1,1]], dtype=tf.float32)

#Added a constant d
d = tf.constant(3, dtype=tf.float32)

#Adding c to a and b
ex1 = (a + b + c)

#Multiplying d to the addition of a, b and c.
ex2 = ex1 * d

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

#Printing a+b+c
print(sess.run(ex1, {a: [[1,2,3]], b: [[3,2,1]]}))
#Printing (a+b+c)*d
print(sess.run(ex2, {a: [[1,2,3]], b: [[3,2,1]]}))

eval_ex1 = tf.equal(sess.run(ex1, {a: [[1,2,3]], b: [[3,2,1]]}), [[5., 5., 5.]])
eval_ex2 = tf.equal(sess.run(ex2, {a: [[1,2,3]], b: [[3,2,1]]}), [[15., 15., 15.]])

eval_ex1_accuracy = tf.reduce_mean(tf.cast(eval_ex1, tf.float32))
eval_ex2_accuracy = tf.reduce_mean(tf.cast(eval_ex2, tf.float32))

if(sess.run(eval_ex1_accuracy)== 1.0):
    print("Validated ex1")
else:
    print("ex1 result does not match")

if(sess.run(eval_ex2_accuracy)== 1.0):
    print("Validated ex2")
else:
    print("ex2 result does not match")

