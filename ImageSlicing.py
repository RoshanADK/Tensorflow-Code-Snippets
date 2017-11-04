# Lets make an use of matplotlib

import matplotlib.image as mp_image
import matplotlib.pyplot as plt 
import tensorflow as tf

# Lets read using imread() function
filename = "TensorImage.jpg"                 # Take one image from the same directory or give relative path to an image 
input_image = mp_image.imread(filename)


# Lets first Slice the image that we have just read
# Sliced portion is also image so we need a placeholder to store all the values of slice
sliced_image_placeholder = tf.placeholder("uint8",[None,None,3])

slice = tf.slice(sliced_image_placeholder,[300,0,0],[190,-1,-1])    # Slicing the image >> 190 is Y axis value and 300 being the Region of Interest ROI

with tf.Session() as sess:                                                         # TF has to be executed through Sessions
	result = sess.run(slice,feed_dict={sliced_image_placeholder: input_image})     # sliced portion is being run under session
	print(result.shape)

# plt.imshow(input_image)       # This is the original image
plt.imshow(result)   # If there are two imshow function , last one only gets executed so only result image is displayed
plt.show()