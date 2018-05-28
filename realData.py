import dga_classifier.data as data
from keras.layers.core import Dense
from keras.layers import Dropout
from keras.models import Sequential
from keras.layers.embeddings import Embedding
import cPickle as pickle
import sklearn
from sklearn import feature_extraction
from sklearn.cross_validation import train_test_split

import dga_classifier.bigram as bigram
from keras.models import load_model

if __name__ == "__main__":
    realX = pickle.load(open('realtopdomain.pkl'))
    realY = pickle.load(open('realY.pkl'))
    model = load_model('bigramMode.h5')

    pre_real = model.predict_classes(realX.todense())
    accuracy = sklearn.metrics.accuracy_score(realY, pre_real)
    print '\n real data accuracy: %f' %accuracy