import dga_classifier.data as data
import numpy as np
from keras.layers.core import Dense
from keras.layers import Dropout, Flatten
from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.layers import Conv1D, MaxPooling1D

import sklearn
from sklearn import feature_extraction
from sklearn.cross_validation import train_test_split


model = Sequential()
model.add(Embedding(1000, 64, input_length=10))
# the model will take as input an integer matrix of size (batch, input_length).
# the largest integer (i.e. word index) in the input should be no larger than 999 (vocabulary size).
# now model.output_shape == (None, 10, 64), where None is the batch dimension.

input_array = np.random.randint(1000, size=(32, 10))

model.compile('rmsprop', 'mse')
output_array = model.predict(input_array)
assert output_array.shape == (32, 10, 64)