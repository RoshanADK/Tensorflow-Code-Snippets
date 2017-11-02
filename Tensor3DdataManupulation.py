import numpy as np 

tensor_3D = np.array([[[1,2], [3,4]], [[5,6], [7,8]], [[5,5],[7,8]]])

print(tensor_3D.shape)
#print(tensor_3D.ndim)
#print(tensor_3D.dtype)
#print(tensor_3D[2][1][1])

#lets convert array into tensors
import tensorflow as tf 
tensor3D = tf.convert_to_tensor(tensor_3D, dtype=tf.int32)

with tf.Session() as sess:
	print(sess.run(tensor3D))
