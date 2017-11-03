# Lets do some Gradient Calculation - Differential Equations

# Let y=2x^2 (2x square) . Compute the gradient di y with respect to x=1

import tensorflow as tf 

# Creating a placeholder for the independent variable of the function
x = tf.placeholder(tf.float32)

# Building the function
y = 1 / (x * x)

# This is the available function with x and y as parameters
var_grad = tf.gradients(y,x)

with tf.Session() as sess:
	var_grad_result = sess.run(var_grad, feed_dict={x:2})
	print(var_grad_result)
