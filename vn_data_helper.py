import csv

from sklearn.utils import shuffle
import numpy as np
import os
from io import open

class data_helper():
	def __init__(self, sequence_max_length=1024):
		self.alphabet = 'abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:’"/|_#$%ˆ&*˜‘+=<>()[]{} '
		#print(len(self.alphabet))
		self.char_dict = {}
		self.sequence_max_length = sequence_max_length
		for i,c in enumerate(self.alphabet):
			self.char_dict[c] = i+1

	def char2vec(self, text):
		data = np.zeros(self.sequence_max_length)
		for i in range(0, len(text)):
			if i >= self.sequence_max_length:
				return data
			elif text[i] in self.char_dict:
				data[i] = self.char_dict[text[i]]
			else:
				# unknown character set to be 68
				data[i] = 68
		return data

	def load_txt_file(self, path, num_classes_):
		"""
		Load CSV file, generate one-hot labels and process text data as Paper did.
		"""
		all_data = []
		labels = []

		all_foders = os.listdir(path)
		num_classes = len(all_foders)
		for i, category in enumerate(all_foders):
			category_dir = path+category
			print("Read from %s " % category_dir)
			count = 0
			files = os.listdir(category_dir)
			for filename in files:
				count += 1
				with open(os.path.join(category_dir, filename),"r",encoding="utf-16") as f:
					text = f.read()
					all_data.append(self.char2vec(text))
					one_hot = np.zeros(num_classes)
					one_hot[i] = 1
					labels.append(one_hot)
				# if count>len(files)/100:
				# 	#todo
				# 	break
		return np.array(all_data), np.array(labels)

	def load_test_data(self,datasetpath):
		test_data,test_label = self.load_txt_file(datasetpath+'test/',num_classes_=7)
		return test_data,test_label

	def load_dataset(self, dataset_path, validation_split = 0.1):
		# Read Classes Info
		with open(dataset_path+"classes.txt") as f:
			classes = []
			for line in f:
				classes.append(line.strip())
		f.close()
		num_classes = len(classes)
		# Read CSV Info
		train_data, train_label = self.load_txt_file(dataset_path+'train/', num_classes)
		size = len(train_label)
		vld_index = int(size * validation_split)
		dev_data,dev_label = train_data[:vld_index],train_label[:vld_index]
		train_data,train_label = train_data[vld_index:																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																				],train_label[vld_index:]
		return train_data, train_label, dev_data, dev_label

	def batch_iter(self, data, batch_size, num_epochs, shuffle=True):
		"""
		Generates a batch iterator for a dataset.
		"""
		data = np.array(data)
		data_size = len(data)
		num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
		for epoch in range(num_epochs):
			# Shuffle the data at each epoch
			if shuffle:
				shuffle_indices = np.random.permutation(np.arange(data_size))
				shuffled_data = data[shuffle_indices]
			else:
				shuffled_data = data
			for batch_num in range(num_batches_per_epoch):
				start_index = batch_num * batch_size
				end_index = min((batch_num + 1) * batch_size, data_size)
				yield shuffled_data[start_index:end_index]

if __name__ =="__main__":
	a = data_helper()
	x = a.load_txt_file("dataset/test/", 7)
	print(x)