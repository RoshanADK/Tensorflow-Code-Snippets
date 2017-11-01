# Lets make an use of matplotlib

import matplotlib.image as mp_image

# Lets read using imread() function
filename = "TensorImage.jpg"
input_image = mp_image.imread(filename)

# Rank(Ndim) and shape will be calculated as

print("Input dime = {}". format(input_image.ndim))
print("Input shape = {}". format(input_image.shape))

import matplotlib.pyplot as plt 
plt.imshow(input_image)
plt.show()