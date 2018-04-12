# @author	huizhan
'''
	Simple demo based on paper https://arxiv.org/abs/1507.05717
		'An End-to-End Trainable Neural Network for Image-based 
		Sequence Recognition and Its Application to Scene Text 
		Recognition'
	Thanks a lot.
'''

import tensorflow as tf
from tensorflow.contrib.keras.python.keras import backend as K


def weight(shape):
	init = tf.random_normal_initializer(mean=0.0, stddev=0.01)
	return tf.get_variable('W', shape=shape, initializer=init)


def bias(shape):
	init = tf.zeros_initializer()
	return tf.get_variable('b', shape=shape, initializer=init)


def beta(channels):
	init = tf.constant_initializer(0.0, tf.float32)
	return tf.get_variable('beta', channels, initializer=init)


def gamma(channels):
	init = tf.constant_initializer(1.0, tf.float32)
	return tf.get_variable('gamma', channels, initializer=init)


def conv2d(inputs, W, conv_pad):
	return tf.nn.conv2d(inputs, W, strides=[1,1,1,1], padding=conv_pad)


def maxpool(inputs, size):
	h, w = size
	return tf.nn.max_pool(inputs, ksize=[1,h,w,1], strides=[1,h,w,1], padding='SAME')


def leaky_relu(h, alpha=1/3.0):
	return K.relu(h, alpha=alpha)


def conv_layer(inputs, shape, conv_pad='SAME', name='fc'):
	with tf.variable_scope(name):
		W = weight(shape)
		b = bias([shape[-1]])
		h = leaky_relu(conv2d(inputs, W, conv_pad) + b)
		return h


def fc_layer(inputs, output, name):
	with tf.variable_scope(name):
		shape = tf.shape(inputs)
		n, w, c = shape[0], shape[1], shape[2]
		inputs = tf.reshape(inputs, [n*w, c])
		W = weight([512, output])
		b = bias([output])
		h = leaky_relu(tf.matmul(inputs, W) + b)
		h = tf.reshape(h, [n, w, output])
		return h


def batch_normalize(inputs, channels, name):
	with tf.variable_scope(name):
		e = 0.001
		m, s = tf.nn.moments(inputs, axes=[0,1,2])
		b = beta(channels)
		g = gamma(channels)
		bn = tf.nn.batch_normalization(inputs, m, s, b, g, e)
		return bn


def bilstm(inputs, name):
	with tf.variable_scope(name):
		hidden = 256
		lstm_fw_cell = tf.contrib.rnn.LSTMCell(hidden, state_is_tuple=True)
		lstm_bw_cell = tf.contrib.rnn.LSTMCell(hidden, state_is_tuple=True)
		lstm_out, last_state = tf.nn.bidirectional_dynamic_rnn( \
			lstm_fw_cell, lstm_bw_cell, inputs, dtype=tf.float32)
		lstm_out = tf.concat(lstm_out, axis=-1)
		return lstm_out


def network(label_count):

	output_length = label_count
	x = tf.placeholder(tf.float32, shape=[None, 32, 160, 1], name='x')
	y_indices = tf.placeholder(tf.int64, name='y_indices')
	y_values = tf.placeholder(tf.int32, name='y_values')
	y_shape = tf.placeholder(tf.int64, name='y_shape')
	y_ = tf.SparseTensor(y_indices, y_values, y_shape)
	seq_len = tf.placeholder(tf.int32, shape=[None], name='seq_len')
	learning_rate = 0.005

	feed = {'x':x, 'y_indices':y_indices, 'y_values':y_values, \
			'y_shape':y_shape, 'seq_len':seq_len}

	#32x160x1
	net = conv_layer(x, [3,3,1,64], name='conv1') 		#32x160x64
	net = maxpool(net, [2,2])							#16x80x64
	net = conv_layer(net, [3,3,64,128], name='conv2')	#16x80x128
	net = maxpool(net, [2,2])							#8x40x128
	net = conv_layer(net, [3,3,128,256], name='conv3')	#8x40x256
	net = conv_layer(net, [3,3,256,256], name='conv4')	#8x40x256
	net = maxpool(net, [2,1])							#4x40x256
	net = conv_layer(net, [3,3,256,512], name='conv5')	#4x40x512
	net = batch_normalize(net, 512, name='bn1')			#4x40x512
	net = conv_layer(net, [3,3,512,512], name='conv6')	#4x40x512
	net = batch_normalize(net, 512, name='bn2')			#4x40x512
	net = maxpool(net, [2,1])							#2x40x512
	net = conv_layer(net, [2,2,512,512], name='conv7', conv_pad='VALID')
														#1x39x512
	net = tf.squeeze(net, [1])
	net = bilstm(net, name='blstm1')
	net = bilstm(net, name='blstm2')
	net = fc_layer(net, output_length, name='fc1')

	loss = tf.nn.ctc_loss(y_, net, seq_len, time_major=False)
	cost = tf.reduce_mean(loss)

	# train_step = tf.train.AdamOptimizer(learning_rate).minimize(cost)
	# train_step = tf.train.AdadeltaOptimizer(learning_rate, 0.9, 1e-8).minimize(cost)
	# train_step = tf.train.MomentumOptimizer(learning_rate, 0.9).minimize(cost)
	# train_step = tf.train.RMSPropOptimizer(learning_rate, 0.9).minimize(cost)
	train_step = tf.train.MomentumOptimizer(learning_rate, 0.9, use_nesterov=True).minimize(cost)

	# transpose to time major
	net = tf.transpose(net, (1, 0, 2))
	decoded, log_prob = tf.nn.ctc_greedy_decoder(net, seq_len)
	ler = tf.reduce_mean(tf.edit_distance(tf.cast(decoded[0], tf.int32), y_))
	decoded = tf.sparse_tensor_to_dense(decoded[0], default_value=0)

	return feed, cost, train_step, decoded, ler

