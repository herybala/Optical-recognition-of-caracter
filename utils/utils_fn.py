import os
import tensorflow as tf

batch_size = 1120
num_of_threads=16
min_after_dequeue=10000
capacity = min_after_dequeue+(num_of_threads+1)*batch_size

def read_data(file_list):
	'''
	read data from tfrecords file
	'''
	file_queue=tf.train.string_input_producer(file_list)
	feature = {'images': tf.FixedLenFeature([], tf.string),'labels': tf.FixedLenFeature([], tf.string)}
	reader = tf.TFRecordReader()
	_,record=reader.read(file_queue)#read a record
	features = tf.parse_single_example(record, features=feature)
	img = tf.decode_raw(features['images'], tf.uint8)
	label = tf.decode_raw(features['labels'], tf.uint8)
	return img,label

def minibatch(batch_size, filename, file_count, img_size, max_char, class_count):
	'''
	create minibatch
	'''
	file_list=[os.path.join(filename + '%d.tfrecords' % i) for i in range(1, file_count+1)]
	img, label=read_data(file_list)
	img = tf.cast(tf.reshape(img,img_size), dtype = tf.float32)
	label = tf.reshape(label[0], [1, max_char])# added [0] as workaround, need to resolve the issue
	label = tf.one_hot(label[0],class_count,axis=1)# added [0] as workaround, need to resolve the issue
	label = tf.reshape(label,tf.shape(label)[1:])
	img_batch,label_batch= tf.train.shuffle_batch([img, label],batch_size,capacity,min_after_dequeue,num_threads=num_of_threads)
	return img_batch, tf.cast(label_batch, dtype = tf.int64)

def variable(name,shape,initializer,weight_decay = None):
	'''
	create parameter tensor
	'''
	var = tf.get_variable(name, shape, initializer = initializer)
	if weight_decay is not None:
		weight_loss = tf.multiply(tf.nn.l2_loss(var),weight_decay, name="weight_loss")
		tf.add_to_collection('losses', weight_loss)
	return var

#need to customize activation and lrn
def conv_block(block_num,
	       input_data,
	       weights,
	       weight_initializer=tf.contrib.layers.xavier_initializer(),
	       bias_initializer=tf.constant_initializer(0.0),
	       conv_op=[1,1,1,1],
	       conv_padding='SAME',
	       weight_decay=None,
	       lrn=True,
	       dropout=1.0,
	       activation=True):
	'''
	convolutional block
	'''
	with tf.variable_scope('conv'+ str(block_num), reuse = tf.AUTO_REUSE) as scope:
		input_data = tf.nn.dropout(input_data, dropout)
		kernel = variable('weights', weights, initializer = weight_initializer, weight_decay = weight_decay)
		biases = variable('biases', weights[3], initializer=bias_initializer, weight_decay=None)
		conv = tf.nn.conv2d(input_data, kernel, conv_op, padding=conv_padding)
		pre_activation = tf.nn.bias_add(conv, biases)
		if lrn==True:
			pre_activation = tf.nn.lrn(pre_activation, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,name='norm')
		if activation:
			conv_out = tf.nn.relu(pre_activation, name=scope.name)
			return conv_out
		else:
			return pre_activation

def dense_block(block_num,
		input_data,
		neurons,
		weight_initializer=tf.contrib.layers.xavier_initializer(),
		bias_initializer=tf.constant_initializer(0.0),
		weight_decay=None,
		activation=True,
		dropout=1.0):
	'''
	fully connected block
	'''
	with tf.variable_scope('dense'+ str(block_num), reuse = tf.AUTO_REUSE) as scope:
		input_data = tf.nn.dropout(input_data, dropout)
		weights = variable('weights', [input_data.shape[1], neurons], initializer=weight_initializer, weight_decay = weight_decay)
		biases = variable('biases', [1,neurons], initializer = bias_initializer, weight_decay = None)
		dense = tf.matmul(input_data,weights)+biases
		if activation:
			dense=tf.nn.relu(dense, name=scope.name)
		return dense

def loss(logits,labels):
	loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits,labels=labels),name='cross_entropy_loss_mean')
	tf.add_to_collection('losses', loss)
	total_loss=tf.add_n(tf.get_collection('losses'), name='total_loss')
	tf.add_to_collection('losses', total_loss)
	return total_loss

def parameter_update(loss, learning_rate):
	'''
	optimization and parameter update using adam optimizer
	'''
	optimizer=tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
	for var in tf.trainable_variables():
		tf.summary.histogram(var.op.name, var)
	return optimizer

def accuracy_calc(output, label_batch):
	'''
	calculate accuracy
	'''
	correct_prediction = tf.equal(tf.cast(tf.argmax(output, 1),dtype=tf.int32),tf.cast(tf.argmax(label_batch, 1),dtype=tf.int32))
	accuracy=tf.reduce_mean(tf.cast(correct_prediction,"float"))
	return accuracy
