from keras.layers import Conv2D
from keras.models import Sequential
from keras.optimizers import SGD

#image size = 28 x 28, colored , 3 channels

"""
basic 1 convolutional layer:
1 filter
kernel_size / filter_size = (3, 3) 
activation = ReLU

"""

def model_definition():

# feature extraction part

	#define model
	model = Sequential()
	model.add(Conv2D(1, (3, 3), activation = 'relu', input_shape = (28, 28, 3)))

	#compile the model
	opt = SGD(lr = 0.001, momentum = 0.9)
	model.compile(optimizer = opt, losses = "categorical_crossentropy", metrics = ['accuracy'])

	return model


def run_process():
	model = model_definition()
	model.save('example1.h5')
