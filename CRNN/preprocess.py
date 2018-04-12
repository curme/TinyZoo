# @author 	huizhan

import os
import cv2
import json
import numpy
import pandas
import tensorflow as tf
from random import randint, choice
from PIL import Image, ImageFont, ImageDraw



FONTS = []
FONT_ROOT = './Static/Font/'
FILES = {	'fangsong':	'fangsong.ttf',
			'heiti':	'heiti.ttc',
			'kaiti':	'kaiti.ttc',
			'lishu':	'lishu.ttc',
			'songti':	'songti.ttc',
			'xihei':	'xihei.ttc',
			'yahei':	'yahei.ttf',
			'yuanti':	'yuanti.ttc',
			'zhongsong':'zhongsong.ttf'}



# load fonts
font_size = 32
for font_name in FILES.keys():
	font_path = FONT_ROOT + FILES[font_name]
	font = ImageFont.truetype(font_path, font_size)
	FONTS.append(font)



# load chinese character
CHAR_ROOT = './Static/Char/'
FILE = 'class_list.json'
with open(CHAR_ROOT+FILE, 'r+') as f:
	CHINESE = json.load(f)
CHINESE = {id:char for char,id in CHINESE.items()}


# load char map
# PLEASE MANUALLY CONFIGURE THE CHARS HERE
chars = []
char_id_map, id_char_map = {}, {}
for i in range(10): chars.append(str(i))
for i in range(97, 97+26): chars.append(chr(i))
for i in range(65, 65+26): chars.append(chr(i))
for i in range(100): chars.append(CHINESE[i])
for i in range(len(chars)): char_id_map[chars[i]] = i
id_char_map = {v:k for k,v in char_id_map.items()}



def random_string(length):
	'''
		generate random content string
	'''
	string = ''
	for _ in range(length):
		string += id_char_map[randint(1,len(id_char_map)-1)]
	return string



def print_string(font, string):
	'''
		convert the text string to single line picture
	'''
	# make a draw board
	height = max([font.getsize(i)[1] for i in string])
	width = sum([font.getsize(i)[0] for i in string])
	board_size = (width, height)
	board = Image.new('RGB', board_size, (255, 255, 255))
	draw = ImageDraw.Draw(board)
	# draw the strinf
	draw.text((0, 0), string, font=font, fill=0)
	return board



def resize(image, size=(160, 32)):
	'''
		resize the picture to a given size
	'''
	image = image.resize(size,Image.ANTIALIAS)
	return image



def label(name):
	'''
		convert the chars to id labels
	'''
	label_seq = []
	for char in name: 
		label_seq.append(char_id_map[char])
	return label_seq



def dense_to_sparse(sequences):
	'''
		convert the dense type matrix to sparse
	'''
	indices = []
	values = []
	for n, seq in enumerate(sequences):
		indices.extend(zip([n]*len(seq), range(len(seq))))
		values.extend(seq)
	indices = numpy.asarray(indices, dtype=numpy.int64)
	values = numpy.asarray(values, dtype=numpy.int32)
	shape = numpy.asarray([len(sequences), numpy.asarray(indices).max(0)[1]+1], dtype=numpy.int64)
	return {"indices": indices, "values": values, "shape": shape, "sequences": sequences}



def generate_image_batch(size):
	'''
		generate a image batch
	'''
	numbers = []
	files = []

	for _ in range(size):

		font = choice(FONTS)
		number = random_string(10)
		image = print_string(font, number)
		image = image.convert('L')
		image = resize(image)
		image = numpy.array(image)
		image = image.reshape([32, 160, -1])

		numbers.append(number)
		files.append(image)

	data = pandas.DataFrame({'x':files, 'y_':numbers})
	return data



def get_batch(size):
	'''
		prepare a batch for training
	'''
	data = generate_image_batch(size)
	for i in range(len(data)): data.loc[i, 'y_'] = label(data.loc[i, 'y_'])
	batch_x = numpy.array(list(data['x']))
	batch_y = numpy.array(list(data['y_']))
	batch_y = dense_to_sparse(batch_y)
	batch = [batch_x, batch_y]

	return batch

get_batch(1)