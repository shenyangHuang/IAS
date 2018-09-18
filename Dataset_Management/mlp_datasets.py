from keras.datasets import cifar10
from keras.datasets import cifar100
from keras.datasets import mnist
from keras.datasets import fashion_mnist
from keras import utils
import random
import numpy as np


class DataSet(object):
	def __init__(self,train_images,train_labels,
				validation_images,validation_labels,
				test_images,test_labels):


		self._train_images = train_images
		self._train_labels = train_labels
		self._validation_images = validation_images
		self._validation_labels = validation_labels
		self._test_images = test_images
		self._test_labels = test_labels


	@property
	def train_images(self):
		return self._train_images

	@property
	def train_labels(self):
		return self._train_labels

	@property
	def validation_images(self):
		return self._validation_images

	@property
	def validation_labels(self):
		return self._validation_labels

	@property
	def test_images(self):
		return self._test_images

	@property
	def test_labels(self):
		return self._test_labels

	def set_train_images(self,train_images):
		self._train_images = train_images

	def set_train_labels(self,train_labels):
		self._train_labels = train_labels

	def set_validation_images(self,validation_images):
		self._validation_images = validation_images

	def set_validation_labels(self,validation_labels):
		self._validation_labels = validation_labels

	def set_test_images(self,test_images):
		self._test_images = test_images

	def set_test_labels(self,test_labels):
		self._test_labels = test_labels

	def concatenate(self,indicator,new_images,new_labels):
		if(indicator == "train"):
			self._train_images = np.concatenate((self._train_images,new_images),axis=0)
			self._train_labels = np.concatenate((self._train_labels,new_labels),axis=0)
			self.shuffle(indicator)
		elif(indicator == "validation"):
			self._validation_images = np.concatenate((self._validation_images,new_images),axis=0)
			self._validation_labels = np.concatenate((self._validation_labels,new_labels),axis=0)
			self.shuffle(indicator)
		elif(indicator == "test"):
			self._test_images = np.concatenate((self._test_images,new_images),axis=0)
			self._test_labels = np.concatenate((self._test_labels,new_labels),axis=0)
			self.shuffle(indicator)
		else:
			raise ValueError('can only concatenate train, val or test set')

	def shuffle(self,indicator):
		if(indicator == "train"):
			order = random.sample(range(0,len(self._train_images)),len(self._train_images))
			shuffle_images=[]
			shuffle_labels=[]
			for i in range(0,len(self._train_images)):
				shuffle_images.append(self._train_images[order[i]])
				shuffle_labels.append(self._train_labels[order[i]])
			self._train_images = np.asarray(shuffle_images)
			self._train_labels = np.asarray(shuffle_labels)
		elif(indicator == "validation"):
			order = random.sample(range(0,len(self._validation_images)),len(self._validation_images))
			shuffle_images=[]
			shuffle_labels=[]
			for i in range(0,len(self._validation_images)):
				shuffle_images.append(self._validation_images[order[i]])
				shuffle_labels.append(self._validation_labels[order[i]])

			self._validation_images = np.asarray(shuffle_images)
			self._validation_labels = np.asarray(shuffle_labels)
		elif(indicator == "test"):
			order = random.sample(range(0,len(self._test_images)),len(self._test_images))
			shuffle_images=[]
			shuffle_labels=[]
			for i in range(0,len(self._test_images)):
				shuffle_images.append(self._test_images[order[i]])
				shuffle_labels.append(self._test_labels[order[i]])

			self._test_images = np.asarray(shuffle_images)
			self._test_labels = np.asarray(shuffle_labels)
		else:
			raise ValueError('can only shuffle train, val or test set')


def create_2d(num_classes):
	temp=[]
	for i in range(0,num_classes):
		temp.append([])
	return temp



#set up MNIST like datasets (MNIST and Fashion MNIST) for simple Incremental Learning Setting 
#now use randomized order 
def Incremental_MNISTlike(data_str, val_split,order,normalization="numerical"):
	if(data_str == "MNIST"):
		(x_train, y_train), (x_test, y_test) = mnist.load_data()
	elif(data_str == "Fashion"):
		(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
	else:
		raise ValueError('Requested MNIST like dataset is not implemented yet')


	num_classes = 10

	x_train = x_train.reshape(60000, 784)
	x_test = x_test.reshape(10000, 784)

	if(normalization == "numerical"):
		x_train = (x_train / 128 -1)
		x_test = (x_test / 128 -1)
	elif(normalization == "std"):
		S = np.std(x_train,axis=0)
		M = np.mean(x_train,axis=0)
		for i in range(0,len(S)):
			if (S[i] == 0):
				S[i] = 1.0
		x_train = (x_train - M) / S
		x_test = (x_test - M) / S
	else:
		x_train = x_train / 255
		x_test = x_test / 255

	training_images = create_2d(num_classes)
	training_labels = create_2d(num_classes)

	for i in range(0,len(y_train)):
		label = y_train[i]
		training_images[label].append(x_train[i])
		training_labels[label].append(y_train[i])
	#switching order
	switch_images = create_2d(num_classes)
	switch_labels = create_2d(num_classes)
	for i in range(0,len(order)):
		switch_images[i] = training_images[order[i]]
		switch_labels[i] = [i]*len(training_labels[order[i]])
	training_images = switch_images
	training_labels = switch_labels


	testing_images = create_2d(num_classes)
	testing_labels = create_2d(num_classes)
	for i in range(0,len(y_test)):
		label = y_test[i]
		testing_images[label].append(x_test[i])
		testing_labels[label].append(y_test[i])
	#switching order
	switch_images = create_2d(num_classes)
	switch_labels = create_2d(num_classes)
	for i in range(0,len(order)):
		switch_images[i] = testing_images[order[i]]
		switch_labels[i] = [i]*len(testing_labels[order[i]])
	testing_images = switch_images
	testing_labels = switch_labels

	return split_classes(num_classes,training_images,training_labels,testing_images,testing_labels,val_split)


def All_MNISTlike(val_split, normalization="numerical"):
	(x1_train, y1_train), (x1_test, y1_test) = mnist.load_data()
	(x2_train, y2_train), (x2_test, y2_test) = fashion_mnist.load_data()

	num_classes = 20

	x1_train = x1_train.reshape(60000, 784)
	x1_test = x1_test.reshape(10000, 784)
	x2_train = x2_train.reshape(60000, 784)
	x2_test = x2_test.reshape(10000, 784)

	#let fashion mnist be classes 10-19
	y2_train = y2_train + 10 
	y2_test = y2_test + 10

	if(normalization == "numerical"):
		x1_train = (x1_train / 128 -1)
		x1_test = (x1_test / 128 -1)
		x2_train = (x2_train / 128 -1)
		x2_test = (x2_test / 128 -1)
	elif(normalization == "std"):
		S = np.std(x1_train,axis=0)
		M = np.mean(x1_train,axis=0)
		for i in range(0,len(S)):
			if (S[i] == 0):
				S[i] = 1.0
		x1_train = (x1_train - M) / S
		x1_test = (x1_test - M) / S

		S = np.std(x2_train,axis=0)
		M = np.mean(x2_train,axis=0)
		for i in range(0,len(S)):
			if (S[i] == 0):
				S[i] = 1.0
		x2_train = (x2_train - M) / S
		x2_test = (x2_test - M) / S
	else:
		x1_train = x1_train / 255
		x1_test = x1_test / 255
		x2_train = x2_train / 255
		x2_test = x2_test / 255


	training_images = create_2d(num_classes)
	training_labels = create_2d(num_classes)
	for i in range(0,len(y1_train)):
		label = y1_train[i]
		training_images[label].append(x1_train[i])
		training_labels[label].append(y1_train[i])

	for i in range(0,len(y2_train)):
		label = y2_train[i]
		training_images[label].append(x2_train[i])
		training_labels[label].append(y2_train[i])


	testing_images = create_2d(num_classes)
	testing_labels = create_2d(num_classes)
	for i in range(0,len(y1_test)):
		label = y1_test[i]
		testing_images[label].append(x1_test[i])
		testing_labels[label].append(y1_test[i])
	for i in range(0,len(y2_test)):
		label = y2_test[i]
		testing_images[label].append(x2_test[i])
		testing_labels[label].append(y2_test[i])
	
	return split_classes(num_classes,training_images,training_labels,testing_images,testing_labels,val_split)





def split_classes(num_classes,training_images,training_labels,testing_images,testing_labels,val_split):

	datasets = []
	for i in range(0,num_classes):
		num_labels = i+1
		val_num = int(val_split * len(training_images[i]))
		if(i==0):
			num_labels = 2

		onehot_train = utils.to_categorical(training_labels[i],num_labels)
		onehot_test = utils.to_categorical(testing_labels[i],num_labels)
		x_train = training_images[i][val_num:-1]
		y_train = onehot_train[val_num:-1]
		x_val = training_images[i][0:val_num]
		y_val = onehot_train[0:val_num]
		x_test = testing_images[i]
		y_test = onehot_test

		datasets.append(DataSet(x_train,y_train,x_val,y_val,x_test,y_test))

	return datasets


#sum two datsets and store it in old_index
def sum_data(datasets,new_index,old_index):


	old_label_num = len(datasets[old_index].train_labels[0])
	new_label_num = len(datasets[new_index].train_labels[0])
	if(old_label_num!=new_label_num):
		transform_dataset(datasets,new_index,old_index)

	datasets[old_index].concatenate("train",datasets[new_index].train_images,datasets[new_index].train_labels)
	datasets[old_index].concatenate("validation",datasets[new_index].validation_images,datasets[new_index].validation_labels)
	datasets[old_index].concatenate("test",datasets[new_index].test_images,datasets[new_index].test_labels)

	return datasets

		



#transforms one hot encoding of dataset at old_index
def transform_dataset(datasets, new_index, old_index):
		old_dataset = datasets[old_index]
		new_dataset = datasets[new_index]
		past_num = len(old_dataset.train_labels[0])
		new_num = len(new_dataset.train_labels[0])

		if(past_num > new_num):
			raise ValueError('Incorrect use of transform_dataset function, number of labels in previous dataset must be smaller or equal than current')

		if(past_num < new_num):

			fzero = np.zeros((len(old_dataset.train_labels),new_num-past_num),dtype=(old_dataset.train_labels[0].dtype))
			train_labels = np.concatenate((old_dataset.train_labels,fzero),axis=1)
			train_images = old_dataset.train_images
			old_dataset.set_train_labels(train_labels)

			fzero = np.zeros((len(old_dataset.validation_labels),new_num-past_num),dtype=(old_dataset.validation_labels[0].dtype))
			validation_labels = np.concatenate((old_dataset.validation_labels,fzero),axis=1)
			old_dataset.set_validation_labels(validation_labels)

			fzero = np.zeros((len(old_dataset.test_labels),new_num-past_num),dtype=(old_dataset.test_labels[0].dtype))
			test_labels = np.concatenate((old_dataset.test_labels,fzero),axis=1)
			old_dataset.set_test_labels(test_labels)













