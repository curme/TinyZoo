# @author 	huizhan

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import numpy
import tensorflow as tf

from model import network
from preprocess import get_batch, id_char_map

def assemble_feed(feed, size, batch):
	feed_dict = {}
	feed_dict[feed['x']] = batch[0]
	feed_dict[feed['y_indices']] = batch[1]['indices']
	feed_dict[feed['y_values']] = batch[1]['values']
	feed_dict[feed['y_shape']] = batch[1]['shape']
	feed_dict[feed['seq_len']] = [39]*size
	return feed_dict


if __name__ == '__main__':

	feed, cost, train_step, decoded, ler = network(len(id_char_map)+1)

	with tf.Session() as sess:

		sess.run(tf.global_variables_initializer())

		for i in range(100000):

			size = 10
			batch = get_batch(size)
			feed_dict = assemble_feed(feed, size, batch)
			sess.run(train_step, feed_dict=feed_dict)

			if i%10 == 9:

				size = 10
				batch = get_batch(size)
				feed_dict = assemble_feed(feed, size, batch)

				cost_result, error, result = sess.run([cost, ler, decoded], feed_dict=feed_dict)
				temp = lambda x: '_' if x == len(id_char_map) else id_char_map[x]
				str_raw = [''.join([temp(i) for i in seq]) for seq in batch[1]['sequences']]
				str_decoded = [''.join([temp(i) for i in seq]) for seq in result]
				for j in range(len(result)): print(str_raw[j], str_decoded[j])
				print(i+1, 'cost: ', cost_result, '; error: ', error)
				print('\n\n')
