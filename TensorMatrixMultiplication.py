# Tensorflow  Matrix Multiplication and Matrix Addition

import tensorflow as tf 
import numpy as np 

matrix1 = np.array([(2,2,2), (2,2,2), (2,2,2)], dtype='int32')
matrix2 = np.array([(1,1,1), (1,1,1), (1,1,1)], dtype='int32')
matrix3 = np.array([(2,7,2), (1,4,2), (9,0,2)], dtype='float32')

#Displaying the matrices

print("Matrix 1  = ")
print(matrix1)

print("-------------------------")

print("Matrix 2  = ")
print(matrix2)

print("--------------------------")

print("Matrix 3  =")
print(matrix3)


# Converting them into Tensor Data Structure
matrix1 = tf.constant(matrix1)
matrix2 = tf.constant(matrix2)
matrix3 = tf.constant(matrix3)

# Lets do the operations
matrix_product = tf.matmul(matrix1, matrix2)
matrix_sum = tf.add(matrix1, matrix2)
matrix_det = tf.matrix_determinant(matrix3)

with tf.Session() as sess:
	result1 = sess.run(matrix_product)
	result2 = sess.run(matrix_sum)
	result3 = sess.run(matrix_det)

print("Multiplied")
print(result1)
print("Added")
print(result2)
print("Determinant")
print(result3)
