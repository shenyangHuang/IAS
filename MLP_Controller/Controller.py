from __future__ import print_function
from .per_class import class_accuracy
import keras as K
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras import optimizers
from keras import callbacks
from keras import backend
from keras.datasets import mnist
import numpy as np
import copy

import sys
import math



'''
keep it simple
1. stores network weights, bias and architecture
2. allows network transformations
3. build new models to train and test 

requires a unique name for controller and it is also used for checkpointing purpose
'''

class Controller:

	def __init__(self,randomSeed,name):
		self.weights = []
		self.bias = []
		self.architecture = []
		self.seed = 0
		self.name = name



	def export(self):
		return {'weights':copy.deepcopy(self.weights),
				'bias':copy.deepcopy(self.bias),
				'architecture':copy.deepcopy(self.architecture)}


	def param_import(self,params):
		self.weights = copy.deepcopy(params['weights'])
		self.bias = copy.deepcopy(params['bias'])
		self.architecture = copy.deepcopy(params['architecture'])

	def set_architecture(self,architecture):
		self.architecture = copy.deepcopy(architecture)

	def set_seed(self,seed):
		self.seed = seed 

	def return_architecture(self):
		return self.architecture


	def print_architecture(self):
		print ("the network architecture is currently"+ str(self.architecture))


	def print_shape(self):
		for i in range(0,len(self.weights)):
			print("shape of weight matrix at layer" + str(i))
			print(np.array(self.weights[i]).shape)

		for i in range(0,len(self.bias)):
			bias = self.bias[i]
			if(isinstance(bias,tf.Tensor)):
				bias = bias.eval()
			print("shape of bias matrix at layer" + str(i))
			print(np.array(self.bias[i]).shape)
	

	#randomly maps to a certain size 
	def randomMapping(self,n,q):
		if q < n:
			raise ValueError("reduce number of neurons in a level is not allowed")
		
		base = np.arange(0, n, 1)
		extend = np.random.randint(0,n,size=(q-n))
		return np.append(base,extend)
	
	
	#count occurrence for each element 
	def occurrence(self,mapping,n):
		base = np.zeros(n)
		for i in range(0,n):
			base[i] = np.count_nonzero(mapping == i)
		return base


	

	#widen the specified layer 
	#used for Net2WiderNet
	def WiderNetRemapping(self,input_weights, output_weights, input_bias, extended_size):
		prev_size = input_weights.shape[0]
		original_size = input_weights.shape[1]
		out_size = output_weights.shape[1]

		input_mapping = self.randomMapping(original_size,extended_size)
		input_remap = np.zeros((prev_size,extended_size), dtype="float32")
		for i in range(0,prev_size):
			for j in range(0,extended_size):
				input_remap[i][j] = input_weights[i][input_mapping[j]]
	
		occurrence = self.occurrence(input_mapping,original_size)
		output_remap = np.zeros((extended_size,out_size),dtype="float32")
		for i in range(0, extended_size):
			for j in range(0, out_size):
				output_remap[i][j] = output_weights[input_mapping[i]][j] / occurrence[input_mapping[i]]

		bias_remap = np.zeros(extended_size,dtype="float32")
		for i in range(0, extended_size):
			bias_remap[i] = input_bias[input_mapping[i]] / occurrence[input_mapping[i]]

		return {'input_remap':input_remap, 'output_remap':output_remap, 'bias_remap':bias_remap}



	#adds uniform noise to break symmetry 
	def addNoise(self,input_layer, minV, maxV):
		np.random.seed(self.seed)
		noise = np.random.uniform(low=minV, high=maxV,size=input_layer.shape) 
		noise = noise.astype('f')
		return input_layer + noise


	def Net2WiderNet(self,layer_number,extended_size, noise_level):

		if(layer_number == 0):
			raise ValueError('Net2WiderNet Layer number starts from 1 not 0')

		if(extended_size < self.architecture[layer_number-1]):
			raise ValueError('WiderNet can only add neurons, extended size must be larger than original')

		input_weights = self.weights[layer_number-1]
		output_weights = self.weights[layer_number]
		input_bias = self.bias[layer_number-1]
		

		mapping = self.WiderNetRemapping(input_weights, output_weights, input_bias, extended_size)
		
		self.weights[layer_number-1] = self.addNoise(mapping['input_remap'], noise_level*(-1), noise_level)
		self.weights[layer_number] = self.addNoise(mapping['output_remap'], noise_level*(-1), noise_level)
		self.bias[layer_number-1] = mapping['bias_remap']
		self.architecture[layer_number-1] = extended_size



	def Net2DeeperNet(self,layer_number):
		
		#make an identity layer based on the size of the indicated layer
		num_hidden = self.architecture[layer_number-1]

		identity = np.eye(num_hidden)
		zero_bias = np.zeros([num_hidden],dtype='f')

		self.weights.append(identity)
		temp1_weights = self.weights[layer_number]
		temp2_weights = []
		self.weights[layer_number] = identity

		#shift weights 
		for i in range(layer_number+1,(len(self.weights))):
			#shift all weights down one level
			temp2_weights = self.weights[i]
			self.weights[i] = temp1_weights
			temp1_weights = temp2_weights
		
		#shift bias
		self.bias.append(zero_bias)
		temp1_bias = self.bias[layer_number]
		temp2_bias = []
		self.bias[layer_number] = zero_bias
		for i in range(layer_number+1,(len(self.bias))):
			#shift all bias down one level
			temp2_bias = self.bias[i]
			self.bias[i] = temp1_bias
			temp1_bias = temp2_bias

		#shift architecture
		self.architecture.append(num_hidden)
		temp1_architecture = self.architecture[layer_number]
		temp2_architecture = 0
		self.architecture[layer_number] = num_hidden
		for i in range(layer_number+1,(len(self.architecture))):
			temp2_architecture = self.architecture[i]
			self.architecture[i] = temp1_architecture
			temp1_architecture = temp2_architecture


	#Transform output layer weights and bias to add in more output neurons 
	def outLayer_transform(self,new_num,mean=0.0,std_dev=0.35):
		

		past_num = len(self.bias[-1])
		output_weights = self.weights[-1]
		output_bias = self.bias[-1]

		if(past_num > new_num):

			raise ValueError('transforming output layer to smaller sizes are invalid')


		if(past_num < new_num):

			weights_random = np.random.normal(mean,std_dev,[len(output_weights),(new_num-past_num)])
			output_weights = np.concatenate((output_weights,weights_random),axis=1)
			self.weights[-1] = copy.deepcopy(output_weights)


			bias_random = np.zeros(new_num-past_num)
			output_bias = np.concatenate((output_bias,bias_random))
			self.bias[-1] = copy.deepcopy(output_bias)



	#trains one layer MLP as starting point 
	#if Test is True, then returns test accuracy, else returns validation accuracy 
	def train_Teacher(self,dataset,architecture=[20],
					input_dim=784,output_dim=10, 
					lr=0.001,epochs=100, 
					batch_size=32,random_seed=29,
					mute=False,test=True,dropout=True,class_curve=False):

		#set up random seed 
		from numpy.random import seed
		seed(random_seed)
		from tensorflow import set_random_seed
		set_random_seed(random_seed)


		verbose=0
		if(mute):
			verbose=0
		else:
			verbose=1
		x_train = np.asarray(dataset.train_images)
		y_train = np.asarray(dataset.train_labels)
		x_val = np.asarray(dataset.validation_images)
		y_val = np.asarray(dataset.validation_labels)
		x_test = np.asarray(dataset.test_images)
		y_test = np.asarray(dataset.test_labels)

		model = Sequential()
		model.add(Dense(architecture[0], activation='relu',input_dim=input_dim))
		for i in range(1,len(architecture)):
			model.add(Dense(architecture[i], activation='relu'))
			if(dropout):
				model.add(Dropout(0.5))
		model.add(Dense(output_dim, activation='softmax'))
		rmsprop = optimizers.RMSprop(lr=lr, rho=0.9, epsilon=None, decay=0.0)
		model.compile(optimizer=rmsprop,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

		filepath=self.name + "teacher.best.hdf5"
		#create check point to stop the best parameters and use early_stopping to decide when to stop
		checkpoint = callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=verbose, save_best_only=True, mode='auto')
		early_Stopping = callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=20, verbose=verbose, mode='auto')
		Callbacks = [checkpoint,early_Stopping]

		#show training curve of each class 
		if(class_curve):
			class_accu = class_accuracy()
			Callbacks.append(class_accu)

		model.fit(x_train, y_train, 
					epochs=epochs, batch_size=batch_size, 
					verbose=verbose, callbacks=Callbacks,  
					validation_data=(x_val, y_val), 
					shuffle=True)

		model.load_weights(filepath)
		score = 0
		if(test):
			score = model.evaluate(x_test, y_test, batch_size=batch_size,verbose=verbose)
		else:
			score = model.evaluate(x_val, y_val, batch_size=batch_size,verbose=verbose)
		weights = []
		bias = []
		for i in range(0,len(model.layers)):
			parameters = model.layers[i].get_weights()
			#dropout layers doesn't have weight 
			if(not parameters):
				continue
			weights.append(copy.deepcopy(parameters[0]))
			bias.append(copy.deepcopy(parameters[1]))
		
		self.weights = weights
		self.bias = bias 
		self.architecture = copy.deepcopy(architecture)
		return score


	#execute and train the transformed model
	def execute(self,dataset,input_dim=784,
				output_dim=10,lr=0.001,
				epochs=100,batch_size=32,
				random_seed=29,mute=False,
				test=True,noVal=False,
				log_path="none",dropout=True,class_curve=False):

		#set up random seed 
		from numpy.random import seed
		seed(random_seed)
		from tensorflow import set_random_seed
		set_random_seed(random_seed)


		verbose=0
		if(mute):
			verbose=0
		else:
			verbose=1	
		log = False
		if(log_path != "none"):
			log = True

		#check changes in num_classes and change output layer correspondinglys
		past_dim = len(self.bias[-1])
		if(output_dim > past_dim):
			self.outLayer_transform(output_dim,mean=0.0,std_dev=0.35)

		x_train, y_train = np.asarray(dataset.train_images), np.asarray(dataset.train_labels)
		x_val, y_val = np.asarray(dataset.validation_images), np.asarray(dataset.validation_labels)
		x_test, y_test = np.asarray(dataset.test_images), np.asarray(dataset.test_labels)

		if(noVal):
			x_train = np.concatenate((x_train,x_val), axis=0)
			y_train = np.concatenate((y_train,y_val), axis=0)

		architecture = self.architecture
		model = Sequential()

		#build model
		model.add(Dense(architecture[0], activation='relu',input_dim=input_dim))
		for i in range(1,len(architecture)):
			model.add(Dense(architecture[i], activation='relu'))
			if(dropout):
				model.add(Dropout(0.5))
		model.add(Dense(output_dim, activation='softmax'))

		#set weights and bias
		count = 0
		for i in range(0,len(self.weights)):
			if(dropout):
				if(i>0 and i%2 == 0):
					continue
			combined = []
			combined.append(np.asarray(self.weights[count]))
			combined.append(np.asarray(self.bias[count]))
			count = count + 1
			model.layers[i].set_weights(combined)

		rmsprop = optimizers.RMSprop(lr=lr, rho=0.9, epsilon=None, decay=0.0)

		model.compile(optimizer=rmsprop,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

		filepath=self.name + "execute.best.hdf5"

		checkpoint = callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=verbose, save_best_only=True, mode='auto')
		early_Stopping = callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=20, verbose=verbose, mode='auto')
		Callbacks = [checkpoint,early_Stopping]
		#show training curve of each class 
		if(class_curve):
			class_accu = class_accuracy()
			Callbacks.append(class_accu)

		if(log):
			logCall = callbacks.CSVLogger(log_path, separator=',', append=False)
			Callbacks.append(logCall)

		if(not noVal):
			model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, 
					verbose=verbose, callbacks=Callbacks,  
					validation_data=(x_val,y_val), 
					shuffle=True)
		else:
			model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, 
					verbose=verbose, callbacks=Callbacks,  
					validation_data=(x_test,y_test), 
					shuffle=True)

		model.load_weights(filepath)
		score = 0
		if(test):
			score = model.evaluate(x_test, y_test, batch_size=batch_size,verbose=verbose)
		else:
			score = model.evaluate(x_val, y_val, batch_size=batch_size,verbose=verbose)
		weights = []
		bias = []
		for i in range(0,len(model.layers)):
			parameters = model.layers[i].get_weights()
			#dropout layers doesn't have weight 
			if(not parameters):
				continue
			weights.append(copy.deepcopy(parameters[0]))
			bias.append(copy.deepcopy(parameters[1]))
		
		self.weights = weights
		self.bias = bias 
		return score


	#execute and train the transformed model
	def baseline(self,dataset,input_dim=784,
				output_dim=10,lr=0.001,
				epochs=100,batch_size=32,
				random_seed=29,mute=False,
				test=True,noVal=False,
				upload=False,dropout=True,class_curve=False):

		#set up random seed 
		from numpy.random import seed
		seed(random_seed)
		from tensorflow import set_random_seed
		set_random_seed(random_seed)

		verbose=0
		if(mute):
			verbose=0
		else:
			verbose=1

		x_train, y_train = np.asarray(dataset.train_images), np.asarray(dataset.train_labels)
		x_val, y_val = np.asarray(dataset.validation_images), np.asarray(dataset.validation_labels)
		x_test, y_test = np.asarray(dataset.test_images), np.asarray(dataset.test_labels)

		architecture = self.architecture
		if(noVal):
			x_train = np.concatenate((x_train,x_val), axis=0)
			y_train = np.concatenate((y_train,y_val), axis=0)


		model = Sequential()
		model.add(Dense(architecture[0], activation='relu',input_dim=input_dim))
		for i in range(1,len(architecture)):
			model.add(Dense(architecture[i], activation='relu'))
			if(dropout):
				model.add(Dropout(0.5))
		model.add(Dense(output_dim, activation='softmax'))

		rmsprop = optimizers.RMSprop(lr=lr, rho=0.9, epsilon=None, decay=0.0)

		model.compile(loss='categorical_crossentropy',
              optimizer=rmsprop,
              metrics=['accuracy'])


		filepath=self.name + "baseline.best.hdf5"

		checkpoint = callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=verbose, save_best_only=True, mode='auto')
		early_Stopping = callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=20, verbose=verbose, mode='auto')
		Callbacks = [checkpoint,early_Stopping]
		#show training curve of each class 
		if(class_curve):
			class_accu = class_accuracy()
			Callbacks.append(class_accu)

		if(not noVal):
			model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, 
						verbose=verbose, callbacks=Callbacks,  
						validation_data=(x_val,y_val), shuffle=True)
		else:
			model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, 
						verbose=verbose, callbacks=Callbacks,  
						validation_data=(x_test,y_test), shuffle=True)


		score = 0
		model.load_weights(filepath)

		if(test):
			score = model.evaluate(x_test, y_test, batch_size=batch_size,verbose=verbose)
		else:
			score = model.evaluate(x_test, y_test, batch_size=batch_size,verbose=verbose)

		if(upload):
			weights = []
			bias = []
			for i in range(0,len(model.layers)):
				parameters = model.layers[i].get_weights()
				#dropout layers doesn't have weight 
				if(not parameters):
					continue
				weights.append(copy.deepcopy(parameters[0]))
				bias.append(copy.deepcopy(parameters[1]))
			
			self.weights = weights
			self.bias = bias 
		return score


	#get test accuracy 
	def test(self,dataset,input_dim=784,output_dim=10,lr=0.001,batch_size=32,mute=True):

		verbose=0
		if(mute):
			verbose=0
		else:
			verbose=1

		x_test = np.asarray(dataset.test_images)
		y_test = np.asarray(dataset.test_labels)

		architecture = self.architecture

		model = Sequential()
		model.add(Dense(architecture[0], activation='relu',input_dim=input_dim))
		for i in range(1,len(architecture)):
			model.add(Dense(architecture[i], activation='relu'))
		model.add(Dense(output_dim, activation='softmax'))

		#set weights and bias
		for i in range(0,len(self.weights)):
			combined = []
			combined.append(np.asarray(self.weights[i]))
			combined.append(np.asarray(self.bias[i]))
			model.layers[i].set_weights(combined)

		rmsprop = optimizers.RMSprop(lr=lr, rho=0.9, epsilon=None, decay=0.0)
		model.compile(loss='categorical_crossentropy',
              optimizer=rmsprop,
              metrics=['accuracy'])

		score = model.evaluate(x_test, y_test, batch_size=batch_size,verbose=verbose)
		return score

	#get accuracy for different classes 
	def class_accuracy(self,dataset,input_dim=784,output_dim=10,lr=0.001,batch_size=32,mute=True):
		verbose=0
		if(mute):
			verbose=0
		else:
			verbose=1

		x_test = np.asarray(dataset.test_images)
		y_test = np.asarray(dataset.test_labels)

		architecture = self.architecture

		model = Sequential()
		model.add(Dense(architecture[0], activation='relu',input_dim=input_dim))
		for i in range(1,len(architecture)):
			model.add(Dense(architecture[i], activation='relu'))
		model.add(Dense(output_dim, activation='softmax'))

		#set weights and bias
		for i in range(0,len(self.weights)):
			combined = []
			combined.append(np.asarray(self.weights[i]))
			combined.append(np.asarray(self.bias[i]))
			model.layers[i].set_weights(combined)

		rmsprop = optimizers.RMSprop(lr=lr, rho=0.9, epsilon=None, decay=0.0)
		model.compile(loss='categorical_crossentropy',
              optimizer=rmsprop,
              metrics=['accuracy'])

		predictions = model.predict(x_test,verbose=verbose)

		class_count = np.zeros((output_dim,2))
		for i in range(0,len(predictions)):
			correct = np.argmax(y_test[i])
			class_count[correct][1] = class_count[correct][1] + 1
			if( (np.argmax(predictions[i])) == correct):
				class_count[correct][0] = class_count[correct][0] + 1

		for i in range(0,output_dim):
			print ("class " + str(i+1) + " : " + str(class_count[i][0]) + " / " + str(class_count[i][1]))
			accuracy = float(class_count[i][0]) / float(class_count[i][1])
			print ("test accuracy is " + str(accuracy) )



	#takes in extra input of a curriculum compare to execute
	#curriculum is a mask for the previous dataset only 
	def curriculum_learning(self,prev_dataset,new_dataset,curriculum,input_dim=784,
					output_dim=10,lr=0.001,
					epochs=100,batch_size=32,
					random_seed=29,mute=False,
					test=True,log_path="none",dropout=True):
		verbose=0
		if(mute):
			verbose=0
		else:
			verbose=1	
		
		#check changes in num_classes and change output layer correspondinglys
		past_dim = len(self.bias[-1])
		if(output_dim > past_dim):
			self.outLayer_transform(output_dim,mean=0.0,std_dev=0.35)
		x_train_old = np.asarray(prev_dataset.train_images)
		y_train_old = np.asarray(prev_dataset.train_labels)
		x_val_old = np.asarray(prev_dataset.validation_images)
		y_val_old = np.asarray(prev_dataset.validation_labels)
		x_test_old = np.asarray(prev_dataset.test_images)
		y_test_old = np.asarray(prev_dataset.test_labels)

		x_train_new = np.asarray(new_dataset.train_images)
		y_train_new = np.asarray(new_dataset.train_labels)
		x_val_new = np.asarray(new_dataset.validation_images)
		y_val_new = np.asarray(new_dataset.validation_labels)
		x_test_new = np.asarray(new_dataset.test_images)
		y_test_new = np.asarray(new_dataset.test_labels)


		architecture = self.architecture
		model = Sequential()

		#build model
		model.add(Dense(architecture[0], activation='relu',input_dim=input_dim))
		for i in range(1,len(architecture)):
			model.add(Dense(architecture[i], activation='relu'))
			if(dropout):
				model.add(Dropout(0.5))
		model.add(Dense(output_dim, activation='softmax'))

		#set weights and bias
		count = 0
		for i in range(0,len(self.weights)):
			if(dropout):
				if(i>0 and i%2 == 0):
					continue
			combined = []
			combined.append(np.asarray(self.weights[count]))
			combined.append(np.asarray(self.bias[count]))
			count = count + 1
			model.layers[i].set_weights(combined)
		rmsprop = optimizers.RMSprop(lr=lr, rho=0.9, epsilon=None, decay=0.0)
		model.compile(optimizer=rmsprop,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

		filepath=self.name + "execute.curriculum.hdf5"

		f = open(log_path+"accuracy.txt", "w")
		f.write("epoch,new val accuracy,old val accuracy,overall val accuracy"+'\n')
		# print("epoch,new val accuracy,old val accuracy,overall val accuracy")
		counter = 0
		prev_accu = 0
		best_accu = 0
		patience = 0
		for i in range(0,epochs):
			#if training epochs exceeds curriculum, just recycle again
			if(counter == len(curriculum)):
				counter = 0
			x = []
			y = []
			for j in range(0,len(x_train_old)):
				#only take data samples based on mask
				if(curriculum[i][j] == 1):
					x.append(x_train_old[j])
					y.append(y_train_old[j])
			isEmpty = False
			if(x==[]):
				isEmpty = True
			x = np.asarray(x)
			y = np.asarray(y)
			if(not isEmpty):
				x = np.concatenate((x,x_train_new),axis=0)
				y = np.concatenate((y,y_train_new),axis=0)
			else:
				x = x_train_new
				y = y_train_new

			idx = np.random.permutation(len(x))
			x = x[idx]
			y = y[idx]

			model.train_on_batch(x, y)
			new_accuracy = model.test_on_batch(x_val_new, y_val_new)
			old_accuracy = model.test_on_batch(x_val_old, y_val_old)
			x_overall = np.concatenate((x_val_new,x_val_old),axis=0)
			y_overall = np.concatenate((y_val_new,y_val_old),axis=0)
			overall = model.test_on_batch(x_overall,y_overall)
			# print (str(i)+","+str(new_accuracy[1])+','+str(old_accuracy[1])+','+str(overall[1]) + '\n')
			f.write(str(i)+","+str(new_accuracy[1])+','+str(old_accuracy[1])+','+str(overall[1]) + '\n')
			counter = counter + 1

			#implementing check pointing base on overall accuracy 
			if(overall[1] > best_accu):
				model.save_weights(filepath)
				best_accu = overall[1]
			if(overall[1] < prev_accu):
				patience = patience + 1
			if(overall[1] >= prev_accu):
				patience = 0
			if(patience >= 20):
				break
			if(i > 100):
				#too long 
				break
			prev_accu = overall[1]


		model.load_weights(filepath)
		score = 0
		if(test):
			x_test = np.concatenate((x_test_new,x_test_old),axis=0)
			y_test = np.concatenate((y_test_new,y_test_old),axis=0)
			predictions = model.predict(x_test,verbose=verbose)
			class_count = np.zeros((output_dim,2))
			for i in range(0,len(predictions)):
				correct = np.argmax(y_test[i])
				class_count[correct][1] = class_count[correct][1] + 1
				if( (np.argmax(predictions[i])) == correct):
					class_count[correct][0] = class_count[correct][0] + 1
			for i in range(0,output_dim):
				print ("class " + str(i+1) + " : " + str(class_count[i][0]) + " / " + str(class_count[i][1]))
				accuracy = float(class_count[i][0]) / float(class_count[i][1])
				print ("test accuracy is " + str(accuracy) )
			score = model.evaluate(x_test, y_test, batch_size=batch_size,verbose=verbose)
		else:
			x_val = np.concatenate((x_val_new,x_val_old),axis=0)
			y_val = np.concatenate((y_val_new,y_val_old),axis=0)
			score = model.evaluate(x_val, y_val, batch_size=batch_size,verbose=verbose)
		weights = []
		bias = []
		for i in range(0,len(model.layers)):
			parameters = model.layers[i].get_weights()
			#dropout layers doesn't have weight 
			if(not parameters):
				continue
			weights.append(copy.deepcopy(parameters[0]))
			bias.append(copy.deepcopy(parameters[1]))
		
		self.weights = weights
		self.bias = bias 
		return score

















		
