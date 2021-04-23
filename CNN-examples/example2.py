from keras.layers import Conv2D
from keras.models import Sequential
from keras.optimizers import SGD

#image size = 28 x 28, colored , 3 channels

"""
2 convolutional layers:
32 and 64 filters respectively
kernel initializer = 'he_uniform'
kernel_size / filter_size = (3, 3) 
activation = ReLU

Problem:
Every time the output from the convolutional layer (called as the output feature map),
reduces in size.

It is fine with large images but not for small images.

"""

def model_definition():

# feature extraction part

	#define model
	model = Sequential()
	model.add(Conv2D(32, (3, 3), activation = 'relu', input_shape = (28, 28, 3), kernel_initializer = 'he_uniform'))
	model.add(Conv2D(64, (3, 3), activation = 'relu', kernel_initializer = 'he_uniform'))

	#compile the model
	opt = SGD(lr = 0.001, momentum = 0.9)
	model.compile(optimizer = opt, losses = "categorical_crossentropy", metrics = ['accuracy'])

	return model


def run_process():
	model = model_definition()
	model.save('example2.h5')
