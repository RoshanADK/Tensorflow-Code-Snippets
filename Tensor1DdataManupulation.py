import numpy as np

tensor_1D = np.array([1.3, 1, 4.0, 23.29])

#print(tensor_1D.ndim)  #Number of Dimensions

#print(tensor_1D.shape)  #Shape of the array 

#print(tensor_1D.dtype)  # Data type of the values

# Let's convert the python DS to Tensors
import tensorflow as tf
tf_tensor = tf.convert_to_tensor(tensor_1D, dtype=tf.float64)

with tf.Session() as sess:
	print(sess.run(tf_tensor))
	print(sess.run(tf_tensor[0]))
	print(sess.run(tf_tensor[2]))
