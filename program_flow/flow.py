#imports


#load dataset and one hot encoding
def data_into_memory():
  
  #split data into X_train, y_train & X_test, y_test
  (X_train, y_train), (X_test, y_test) = #loading the dataset
  #followed by one hot encoding with to_categorical() for the dependent variable
  y_train = to_categorical(y_train)
  y_test = to_categorical(y_test)
  
  return X_train, y_train, X_test, y_test


#prepare pixels : normalization for X_train and X_test
def normalize(X_train, X_test):
  
  #pixels values are unsigned integers so convert their data type to 'float32'
  X_train = X_train.astype('float32')
  X_test = X_test.astype('float32')

  #values of pixels for color channels is between 0 and 255
  # divide the arrays by 255.0
  X_train = X_train / 255.0
  X_test = X_test / 255.0

#define model 3 VGG blocks: feature extraction and classfication
def model_definition():
  model = Sequential()
  .
  .
  .

	return model 


#call order:
"""
1. data_into_memory()
2. normalize()
3. model_definition()
"""

	
