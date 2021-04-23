
#imports 
import sys
from keras.datasets import cifar10
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.optimizers import SGD

"""
VGG block:
stacking of convolutional layers followed by MaxPooling
"""

def model_definition1():

	#1 VGG block

	#feature extraction part
	model = Sequential()
	model.add(Conv2D(32, (3, 3), activation = 'relu', kernel_initializer = 'he_uniform', padding = 'same',input_shape = (32, 32, 3)))
	model.add(Conv2D(32, (3, 3), padding = 'same', activation = 'relu', kernel_initializer = 'he_uniform'))
	model.add(MaxPooling2D((2, 2)))


	#classifier part
	model.add(Flatten())
	model.add(Dense(128, activation = 'relu', kernel_initializer = 'he_uniform'))
	model.add(Dense(10, activation = 'softmax'))

	#compiling the model
	opt = SGD(lr = 0.001, momentum = 0.9)
	model.compile(optimizer = opt, loss = 'categorical_crossentropy', metrics = ['accuracy'])

	return model

def model_definition2():

#2 VGG blocks
# feature extraction part
	model = Sequential()

	#block 1: 32 filters
	model.add(Conv2D(32, (3, 3), activation = 'relu', padding = 'same', kernel_initializer = 'he_uniform', input_shape = (32, 32, 3)))
	model.add(Conv2D(32, (3, 2), activation = 'relu', padding = 'same', kernel_initializer = 'he_uniform'))
	model.add(MaxPooling2D((2, 2)))

	#block 2: 64 filters	
	model.add(Conv2D(64, (3, 3), activation = 'relu', padding = 'same', kernel_initializer = 'he_uniform'))
	model.add(Conv2D(64, (3, 3), activation = 'relu', padding = 'same', kernel_initializer = 'he_uniform'))
	model.add(MaxPooling2D((2, 2)))

# classifier part

	model.add(Flatten())
	model.add(Dense(128, activation = 'relu', kernel_initializer = 'he_uniform'))
	model.add(Dense(10, activation = 'softmax'))

#compilation part
	opt = SGD(lr = 0.001, momentum = 0.9)
	model.compile(optimizer = opt, losses = 'categorical_crossentropy', metrics = ['accuracy'])

	return model
