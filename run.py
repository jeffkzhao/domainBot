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

RESULT_FILE = 'results.pkl'

def run_experiments(isn1=True, isn2=True, isn3=True, issigmoid=True, iscnn=True, nfolds=10):
    """Runs all experiments"""
    n1_results = None
    n2_results = None
    n3_results = None
    sigmoid_results = None
    cnn_results = None

    if isn1:
        n1_results = unigram.run(nfolds=nfolds)

    if isn3:
        n3_results = trigram.run(nfolds=nfolds)
    if issigmoid:
        sigmoid_results = sigmoid.run(nfolds=nfolds)

    if isn2:
        n2_results = bigram.run(nfolds=nfolds)
    if iscnn:
	cnn_results = cnn.run(nfolds=nfolds)

    return n1_results, n2_results, n3_results, sigmoid_results, cnn_results

def create_figs(isn1=True, isn2=True, isn3=True, issigmoid=True, iscnn=True, nfolds=10, force=False):
    """Create figures"""
    # Generate results if needed
    if force or (not os.path.isfile(RESULT_FILE)):
        n1_result, n2_result, n3_result, sigmoid_result, cnn_result = run_experiments(isn1, isn2, isn3, issigmoid, iscnn,  nfolds)

        results = {'n1': n1_result, 'n2': n2_result, 'n3': n3_result, 'sigmoid': sigmoid_result, 'cnn': cnn_result}

        pickle.dump(results, open(RESULT_FILE, 'w'))
    else:
        results = pickle.load(open(RESULT_FILE))

    # Extract and calculate bigram ROC
    if results['n1']:
        n1_results = results['n1']
        fpr = []
        tpr = []
        for n1_result in n1_results:
            t_fpr, t_tpr, _ = roc_curve(n1_result['y'], n1_result['probs'])
            fpr.append(t_fpr)
            tpr.append(t_tpr)
        n1_binary_fpr, n1_binary_tpr, n1_binary_auc = calc_macro_roc(fpr, tpr)

    if results['n2']:
        n2_results = results['n2']
        fpr = []
        tpr = []
        for n2_result in n2_results:
            t_fpr, t_tpr, _ = roc_curve(n2_result['y'], n2_result['probs'])
            fpr.append(t_fpr)
            tpr.append(t_tpr)
        n2_binary_fpr, n2_binary_tpr, n2_binary_auc = calc_macro_roc(fpr, tpr)

    if results['n3']:
        n3_results = results['n3']
        fpr = []
        tpr = []
        for n3_result in n3_results:
            t_fpr, t_tpr, _ = roc_curve(n3_result['y'], n3_result['probs'])
            fpr.append(t_fpr)
            tpr.append(t_tpr)
        n3_binary_fpr, n3_binary_tpr, n3_binary_auc = calc_macro_roc(fpr, tpr)

    if results['sigmoid']:
        sigmoid_results = results['sigmoid']
        fpr = []
        tpr = []
        for sigmoid_result in sigmoid_results:
            t_fpr, t_tpr, _ = roc_curve(sigmoid_result['y'], sigmoid_result['probs'])
            fpr.append(t_fpr)
            tpr.append(t_tpr)
        sigmoid_binary_fpr, sigmoid_binary_tpr, sigmoid_binary_auc = calc_macro_roc(fpr, tpr)
"""
    # Save figure
    from matplotlib import pyplot as plt
    with plt.style.context('bmh'):
        #plt.plot(lstm_binary_fpr, lstm_binary_tpr,
        #         label='LSTM (AUC = %.4f)' % (lstm_binary_auc, ), rasterized=True)
        if isn1:
            plt.plot(n1_binary_fpr, n1_binary_tpr,
                    label='unigrams (AUC = %.4f)' % (n1_binary_auc, ), rasterized=True)

        if isn2:
            plt.plot(n2_binary_fpr, n2_binary_tpr,
                    label='bigrams (AUC = %.4f)' % (n2_binary_auc,), rasterized=True)

        if isn3:
            plt.plot(n3_binary_fpr, n3_binary_tpr,
                    label='trigrams (AUC = %.4f)' % (n3_binary_auc,), rasterized=True)
        if issigmoid:
            plt.plot(sigmoid_binary_fpr, sigmoid_binary_tpr,
                    label='randomforest (AUC = %.4f)' % (sigmoid_binary_auc,), rasterized=True)



        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC', fontsize=20)
        plt.legend(loc="lower right", fontsize=12)

        plt.tick_params(axis='both', labelsize=12)
        plt.savefig('results.png')
"""
def calc_macro_roc(fpr, tpr):
    """Calcs macro ROC on log scale"""
    # Create log scale domain
    all_fpr = sorted(itertools.chain(*fpr))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(len(tpr)):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    return all_fpr, mean_tpr / len(tpr), auc(all_fpr, mean_tpr) / len(tpr)

if __name__ == "__main__":
    create_figs(isn1=False, isn2=False, isn3=False, issigmoid=False, nfolds=1) # Run with 1 to make it fast
    #create_figs(nfolds=1)
