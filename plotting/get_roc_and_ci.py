import numpy as np
import pandas as pd
from time import time
from matplotlib import pyplot as plt
from sklearn import preprocessing
import seaborn as sns
from sklearn.pipeline import make_pipeline
# for CV
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.metrics import roc_auc_score
from random import randint
from sklearn import metrics

# plot the ROC curves and CI

"""
A method for calculating and plotting the ROC for each of the model in the given list. 
""" 
def get_roc_and_ci(N, k, ax, iris, my_models):
    
    """
    Takes the dataset and performs the binary classification N times on the target column, 
    then returns the AUC scores and the plot of the ROC.

    Parameters
    ----------
    N : integer
        The number of iterations under which classification is performed

    k: inetger
        Index of the current plot

    ax: integer
        index of the current axis

    iris: data frame
        dataset on which classification is performed

    my_models: dictionary
        list of the models for implementing the classification

    Returns
    -------
    The figure grid with ROC plots for every model in the given list
    
    """
    
    tprs = []
    all_auc = []
    base_fpr = np.linspace(0, 1, 101)
    my_model = my_models[k]['model']

    y = iris['target'].astype('int')
    X = iris.drop(['target'], axis=1)

    for i in range(N):
        print(i)
        s_seed = randint(1, N)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=s_seed)
        model = my_model.fit(X_train, y_train)

        auc = metrics.roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
        fpr, tpr, thresholds = metrics.roc_curve(y_test, model.predict_proba(X_test)[:, 1])
        plt.sca(ax)
        tpr = np.interp(base_fpr, fpr, tpr)
        tpr[0] = 0.0
        tprs.append(tpr)
        all_auc.append(auc)
        #plt.plot(cur_auc, label='%s ROC (area = %0.2f)' % (m['label'], auc))
    # Custom settings for the plot
    # confidence intervals
    tprs = np.array(tprs)
    mean_tprs = tprs.mean(axis=0)
    std = tprs.std(axis=0)
    tprs_upper = np.minimum(mean_tprs + std, 1)
    tprs_lower = mean_tprs - std
    # ploooooooooot
    plt.plot(base_fpr, mean_tprs, 'b',label=' ROC (area = %0.2f)' % (np.mean(all_auc)))
    plt.fill_between(base_fpr, tprs_lower, tprs_upper, color='grey', alpha=0.3)
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([-0.01, 1.01])
    plt.ylim([-0.01, 1.01])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.legend(loc = 'lower right', fontsize = 15)
    
    
def plot_for_every_model(N, iris, dataset_name, my_models):

    fig, axes = plt.subplots(nrows = 3, ncols= 3)
    # for confusion matrix
    k = 0
    for i in range(len(axes)):
        for j in range(len(axes[0])):
            if k == len(my_models):
                break
            
            get_roc_and_ci(N, k, axes[i][j], iris, my_models)
            axes[i][j].title.set_text(my_models[k]['label'])
            # next               
            k = k + 1
    fig.subplots_adjust(wspace=0.3, hspace= 0.5)
    fig.suptitle('ROC score for tested models for %s' % dataset_name, fontsize = 25)
    plt.show()
    
    print('all plots should be printed')


