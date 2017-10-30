import tensorflow as tf 
a = tf.constant(10,name='a')
b = tf.constant(20,name='b')

y = tf.Variable(a+b*2,name='y')

model = tf.global_variables_initializer()

with tf.Session() as session:

	merged = tf.global_summaries_merger()
	writer = tf.train.SummaryWriter("/tmp/tensorflowlogs", sessiion.graph)
	session.run(model)
	print(session.run(y))
 

	
