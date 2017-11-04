
# Sometimes we need to work with Normal Distribution (Gaussian) instead of random uniform numbers

# Normal Distribution has the HILLY Look with more data concentrated at the center
import tensorflow as tf 
import matplotlib.pyplot as plt 

# Function is same as Uniform Distribution , but the parameters differ (100 = Shape , Mean = Center of graph , stddev = Till what range ?)
norm = tf.random_normal([100], mean=0, stddev=5)

# Now Lets have some Say SON :D Get it ? Session -> Say Son ! Wabalaba Dab Dab :D Don't you watch Rick and Morty ? This is RickDICKulous .

with tf.Session() as sess:

	plt.hist(norm.eval(), normed=True)
	plt.show()
