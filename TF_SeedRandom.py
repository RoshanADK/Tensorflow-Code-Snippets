
# This program generates uniform distribution using seeds

import tensorflow as tf

uniform_with_seed = tf.random_uniform([1],seed=1)
uniform_without_seed = tf.random_uniform([1])


print("First Run")
with tf.Session() as first_session:
	print("uniform with seed 1 = {}"\
		.format(first_session.run(uniform_with_seed)))
	print("uniform with seed 1 = {}"\
		.format(first_session.run(uniform_with_seed)))
	print("uniform without seed = {}"\
		.format(first_session.run(uniform_without_seed)))
	print("uniform without seed = {}"\
		.format(first_session.run(uniform_without_seed)))

	print("----------------------------------------------------")

print("Second Run")
with tf.Session() as second_session:
	print("Uniformm with seed 1 = {}"\
		.format(second_session.run(uniform_with_seed)))
	print("Uniformm with seed 1 = {}"\
		.format(second_session.run(uniform_with_seed)))
	print("Uniformm without seed = {}"\
		.format(second_session.run(uniform_without_seed)))
	print("Uniformm without seed = {}"\
		.format(second_session.run(uniform_without_seed)))


# Run the script. We can observe that having seed will always yield same random value which is why we call it pseudo random :D 
# Random numbers just became so weird, they are no more random now !  	
