from keras.layers import Conv2D
from keras.models import Sequential
from keras.optimizers import SGD
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

#image size = 28 x 28, colored , 3 channels

"""
2 convolutional layers:
32 and 64 filters respectively
kernel initializer = 'he_uniform'
kernel_size / filter_size = (3, 3) 
activation = ReLU

classfier part:
Flatten the image
interpret the feature extracted in deep layers
classify them into categories required by your dataset
caluclate their probability distribution with softmax. 

"""

def model_definition():

# feature extraction part

	#define model
	 model = Sequential() model.add(Conv2D(32, (3, 3), activation =
	'relu', padding = 'same', input_shape = (28, 28, 3), kernel_initializer = 'he_uniform'))
	model.add(MaxPooling2D((2, 2)))


	model.add(Conv2D(64, (3, 3), activation = 'relu', padding = 'same', kernel_initializer = 'he_uniform'))
	model.add(MaxPooling2D((2,2)))


# classifier part
	model.add(Flatten())	
	model.add(Dense(100, activation = 'relu', kernel_initializer = 'he_uniform'))
	model.add(Dense(5, activation = 'softmax'))	


	#compile the model
	opt = SGD(lr = 0.001, momentum = 0.9)
	model.compile(optimizer = opt, losses = "categorical_crossentropy", metrics = ['accuracy'])


	return model


def run_process():
	model = model_definition()
	model.save('example6.h5')
