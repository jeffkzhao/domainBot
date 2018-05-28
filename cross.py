"""Run experiments and create figs"""
# -*- coding: utf-8 -*-
import itertools
import os
import pickle
import matplotlib
matplotlib.use('Agg')
import numpy as np

import dga_classifier.unigram as unigram
import dga_classifier.bigram as bigram
import dga_classifier.trigram as trigram
import dga_classifier.sigmoid as sigmoid
import dga_classifier.lstm as lstm
import dga_classifier.cnn as cnn

from scipy import interp
from sklearn.metrics import roc_curve, auc




if __name__ == "__main__":
    bigram.run(nfolds=2)
