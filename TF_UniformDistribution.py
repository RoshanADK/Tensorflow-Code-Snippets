# Working on Uniformly Distributed Random Numbers
# Lets work with Random numbers and try to visualize them using matplotlib library

import tensorflow as tf 
import matplotlib.pyplot as plt

# This function returns 100 Random Floating numbers between 0 and 9 (100 = Shape)
uniform = tf.random_uniform([100],minval=0,maxval=9,dtype=tf.float32)

# As usual creating Session to execute Tensors
with tf.Session() as sess:
	print(uniform.eval())  # Eval function is another way to execute TF
	plt.hist(uniform.eval(), normed = True)    # Visualizing through Histograms
	plt.show()