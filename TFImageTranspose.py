# Lets make an use of matplotlib to Load Image using imread function
import matplotlib.image as mp_image
import matplotlib.pyplot as plt 
import tensorflow as tf

# Lets read using imread() function
filename = "TensorImage.jpg"                 # Take one image from the same directory or give relative path to an image 
input_image = mp_image.imread(filename)

# Lets Perform Some Geometric Transformation here
# We'll start by Transposing the image
# Assign image to Variable x for easy reference
x = tf.Variable(input_image, name="x")

# Now as usual lets initialize the model as
model = tf.global_variables_initializer()

# Transpose Function Invocation
x = tf.transpose(x, perm=[1,0,2])

# Lets now create Session
with tf.Session() as sess:
	sess.run(model)
	result = sess.run(x)

# plt.imshow(input_image)       # This is the original image
plt.imshow(result)   # If there are two imshow function , last one only gets executed so only result image is displayed
plt.show()