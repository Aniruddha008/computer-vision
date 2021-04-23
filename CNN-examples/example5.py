from keras.layers import Conv2D
from keras.models import Sequential
from keras.optimizers import SGD
from keras.layers import MaxPooling2D

#image size = 28 x 28, colored , 3 channels

"""
2 convolutional layers:
32 and 64 filters respectively
kernel initializer = 'he_uniform'
kernel_size / filter_size = (3, 3) 
activation = ReLU

The number of output feature maps increase with the depth of the newtork.
Hence, a 1 x 1 channel wise pooling is introduced, called the projection layer.
Used for reducing dimensions (can be used to increase or decrease output feature maps).


"""

def model_definition():

# feature extraction part

	#define model
	 model = Sequential() model.add(Conv2D(32, (3, 3), activation =
	'relu', padding = 'same', input_shape = (28, 28, 3), kernel_initializer = 'he_uniform'))
	 model.add(Conv2D(32, (1, 1), activation = 'relu'))
	


	model.add(Conv2D(64, (3, 3), activation = 'relu', padding = 'same', kernel_initializer = 'he_uniform'))
	model.add(Conv2D(64, (1, 1), activation = 'relu'))

	#compile the model
	opt = SGD(lr = 0.001, momentum = 0.9)
	model.compile(optimizer = opt, losses = "categorical_crossentropy", metrics = ['accuracy'])

	return model


def run_process():
	model = model_definition()
	model.save('example5.h5')
