import numpy as np

import os
import sys
import re
import random

import yaml
config = yaml.safe_load(open("config.yml"))

import tensorflow as tf

class DataReader(object):
	
	def __init__(self, train_file, valid_file=None, test_file=None, delimiter=None):
		self.delimiter = "\t" if delimiter == None else delimiter
		self.train_data, self.valid_data, self.test_data = self.init_data(train_file, valid_file, test_file)
		self.num_words = sum([len([w for w in l if w != '\n']) for l in self.train_data])
		
		self.i2w, self.w2i = self.make_vocab(self.train_data)
		self.vocab_dim = len(self.i2w)
		print("%d vocabulary" % self.vocab_dim)
		print("%d lines" % len(self.train_data))
		print("%d words" % self.num_words)
	
	def init_data(self, train_file, valid_file, test_file):
		if test_file is not None: # If there is test data, assume all three files are provided (tentative)
			train_data = self.read(train_file)
			valid_data = self.read(valid_file)
			test_data = self.read(test_file)
		elif valid_file is not None: # Else, if there is validation data, assume test data is not available
			train_data = self.read(train_file)
			valid_data = self.read(valid_file)
			test_data = None
		else: # If only training data is provided, shuffle it and randomly split into 90% train, 5% validation and 5% test data
			train_data = self.read(train_file)
			random.shuffle(train_data)
			valid_data = train_data[int(0.9*len(train_data)):int(0.95*len(train_data))]
			test_data = train_data[int(0.95*len(train_data)):]
			train_data = train_data[:int(0.9*len(train_data))]
		
		return train_data, valid_data, test_data
	
	def make_vocab(self, data):
		# Build a vocabulary with words that occur at least vocab_cutoff times
		cutoff = config['words']['vocab_cutoff']
		vocab = {"<s>": cutoff, "<unk>": cutoff}
		for l in data: # Could use a Counter instead
			for w in l:
				if w in vocab:
					vocab[w] += 1
				else:
					vocab[w] = 1
		for w in list(vocab.keys()):
			if vocab[w] < cutoff:
				del vocab[w]
		
		i2w = { i:w for i,w in enumerate(vocab.keys()) }
		w2i = { w:i for i,w in enumerate(vocab.keys()) }
		return i2w, w2i

	def read(self, file):
		def clean(line):
			return [w.strip() for w in line.split(self.delimiter) if len(w.strip()) > 0]
		
		data = []
		# If the input is a directory, traverse its files and assume each file contains one "sentence" worth of data
		if os.path.isdir(file):
			for dirpath, dirs, files in os.walk(file):
				for f in files:
					content = []
					with open(os.path.join(dirpath, f), encoding='utf-8', errors='ignore') as f2:
						for l in f2:
							words = clean(l)
							if len(words) == 0: continue
							content.extend(words)
					if len(content) > 0:
						data.append(content)
		else: # Otherwise, assume each line is a training sample
			with open(file, encoding='utf-8', errors='ignore') as f:
				for l in f:
					words = clean(l)
					if len(words) == 0: continue
					data.append(words)
		return data
	
	def batcher(self, batch_data, is_training=False):
		if is_training: batch_data = random.shuffle(batch_data)
		
		def finish(batch):
			m = max([len(l) for l in batch])
			indices = []
			masks = []
			for l in batch:
				msks = np.array([1.0]*len(l) + [0.0]*(m - len(l)), dtype=np.float32)
				indices.append(l + [0]*(m - len(l)))
				masks.append(msks)
			return tf.constant(indices, dtype="int32"), tf.constant(masks, dtype="float32")
		
		batch = []
		seq = [0]
		max_seq_len = 0
		for l in batch_data:
			for ix, w in enumerate(l):
				seq.append(self.w2i[w if w in self.w2i else "<unk>"])
				if ix == len(l) - 1 or len(seq) > config['data']['words_per_sequence']:
					if len(seq) > max_seq_len:
						max_seq_len = len(seq)
					if max_seq_len*len(batch) > config['data']['batch_size']:
						yield finish(batch)
						batch = []
						batch_len = 0
					batch.append(seq)
					seq = [0]			
		if len(seq) > 1:
			batch.append(seq)
		if len(batch) > 0:
			yield finish(batch)
