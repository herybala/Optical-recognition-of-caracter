import tensorflow as tf
from datetime import datetime as dt
from utils.utils_fn import read_data, minibatch, variable, conv_block, dense_block, loss, parameter_update, accuracy_calc

bold = '\033[1m'
end = '\033[0m'

path = "/home/supernova/Jobs/OCR/begining/ocr_model/"
checkpoint_restore = path + "checkpoints/checkpoint_digi_lc_sel_uc_sel_sign_1.ckpt"
checkpoint_save = path + "checkpoints/checkpoint_digi_lc_sel_uc_sel_sign_1.ckpt"

def inference(image_batch, class_count, weights, dropout=[1,1,1,1],wd=None):
	'''
	Forward propagation
	'''
	i = 0
	conv_op=[[1,1,1,1],[1,1,1,1],[1,1,1,1], [1,1,1,1]]

	conv1 = conv_block(1,image_batch,weights[i], conv_op = conv_op[i], conv_padding='SAME', dropout=dropout[i],weight_decay=wd)
	i=i+1
	pool1=tf.nn.max_pool(conv1, ksize=[1, 4, 4, 1], strides=[1,4,4,1],padding='SAME', name='pool1') #32x32

	conv2 = conv_block(2,pool1,weights[i], conv_op = conv_op[i], conv_padding='SAME', dropout=dropout[i],weight_decay=wd)
	i=i+1
	pool2=tf.nn.max_pool(conv2, ksize=[1, 4, 4, 1], strides=[1,4,4,1],padding='SAME', name='pool2') #8x8

	conv3 = conv_block(3,pool2,weights[i], conv_op = conv_op[i], conv_padding='SAME', dropout=dropout[i],weight_decay=wd)
	i=i+1
	pool3=tf.nn.max_pool(conv3, ksize=[1, 4, 4, 1], strides=[1,4,4,1],padding='SAME', name='pool3') #2x2

	conv4 = conv_block(4,pool3,weights[i], conv_op = conv_op[i], conv_padding='SAME', dropout=dropout[i],weight_decay=wd)
	i=i+1
	pool4=tf.nn.max_pool(conv4, ksize=[1, 2, 2, 1], strides=[1,2,2,1],padding='SAME', name='pool4')#1x1

	flat=tf.reshape(pool4, [tf.shape(image_batch)[0], class_count], name='flat')

	return flat

# Training function

def train(folder_path, train_filename, test_filename,
	  train_data_count, file_count,
	  weights, dropout, wd,
	  img_size, max_char, class_count,
	  batch_size = 32, learning_rate=0.01, epochs=5,
	  restore=False, var_lr = [None,None]):

	train_step = train_data_count//batch_size
	start_time = dt.now()
	#build graph
	with tf.Graph().as_default():
		x_train, y_train = minibatch(batch_size, train_filename, file_count, img_size, max_char, class_count)
		logit_train = inference(x_train, class_count, weights, dropout = dropout, wd = wd)
		cost = loss(logit_train, y_train)
		update=parameter_update(cost,learning_rate)
		accuracy_train = accuracy_calc(logit_train, y_train)
		saver = tf.train.Saver()
		#Start session
		with tf.Session() as sess:
			#initialize the variables
			sess.run(tf.global_variables_initializer())
			sess.run(tf.local_variables_initializer())
			coord = tf.train.Coordinator()
			threads = tf.train.start_queue_runners(coord=coord)
			#restore the variables
			if restore == True:
				loader = tf.train.import_meta_graph(checkpoint_restore +'.meta')
				loader.restore(sess, checkpoint_restore)
			#train for given number of epochs
			for e in range(epochs):
				print(bold + "\nepoch:" + end, e)
				train_epoch_cost = 0
				train_epoch_acc = 0
				#train for given number of steps in one epoch
				for s in range(train_step):
					_,train_batch_cost = sess.run([update, cost])
					train_epoch_cost += train_batch_cost/train_step
				print(bold + "epoch_cost: " + end, train_epoch_cost)
				#calculate accuracy of training set
				for i in range(train_step//5):
					train_batch_acc = sess.run(accuracy_train)
					train_epoch_acc = train_epoch_acc + (train_batch_acc/(train_step//5))
				print(bold + "train epoch accuracy: " + end, train_epoch_acc, "\n")
				#afer every lr[0] epoch decrease learning rate by factor of lr[1]
				if var_lr[0] != None:
					if e%var_lr[0] == 0:
						learning_rate = learning_rate/var_lr[1]
#         if(e%10 == 0 and e!=0):
# # 						save_path = saver.save(sess, checkpoint_save)
# 						print()
			#save all the variables
			print("creating checkpoint...")
			save_path = saver.save(sess, checkpoint_save)
			print("checkpoint created at ", checkpoint_save)
			coord.request_stop()
			coord.join(threads)
			end_time = dt.now()
			print("total time taken =", end_time - start_time)
	return None
