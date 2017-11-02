import numpy as np 

tensor_2D = np.array([(1,2,3,4), (5,6,7,8), (9,10,11,12), (13,14,15,16)])

#print(tensor_2D.ndim)  #Dimension has to be 2

#print(tensor_2D.shape) #Shape must be 4*4

#print(tensor_2D.dtype) #data type 

#print(tensor_2D)
#print(tensor_2D[1][3])

#print(tensor_2D[0:4,0:4])

#Let me convert this multi dimensional array into Tensor now
import tensorflow as tf 

tf_tensor = tf.convert_to_tensor(tensor_2D, dtype=tf.int32)

with tf.Session() as sess:
	print(sess.run(tf_tensor))