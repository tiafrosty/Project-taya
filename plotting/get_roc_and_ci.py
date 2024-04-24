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
def get_roc_and_ci(N, k, ax, iris, my_models, data_from_R):
    
    """
    Takes the dataset and performs the binary classification N times on the target column, 
    then returns the AUC scores together with confidence intervals and the plot of the ROC.

    Parameters
    ----------
    N : integer
        The number of iterations under which classification is performed

    k: integer
        Index of the current plot

    ax: integer
        index of the current axis

    iris: data frame
        dataset on which classification is performed

    my_models: dictionary
        list of the models for implementing the classification

    Returns
    -------
    
    """
    
    # python
    tprs = []
    all_auc = []
    # here we keep predicted targets for all iterations
    all_y_pred = []
    base_fpr = np.linspace(0, 1, 101)
    my_model = my_models[k]['model']

    y = iris['target'].astype('int')
    X = iris.drop(['target'], axis=1)

    for i in range(N):
        #print(i)
        s_seed = randint(1, N)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=s_seed)
        model = my_model.fit(X_train, y_train)
        # try to export the predicted values and plot them together with those obtained in R
        y_pred = model.predict_proba(X_test)[:, 1]

        auc = metrics.roc_auc_score(y_test, y_pred)
        # py
        fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred)

        # py
        tpr = np.interp(base_fpr, fpr, tpr)
        tpr[0] = 0.0
        # py
        tprs.append(tpr)
        all_auc.append(auc)
        all_y_pred.append(y_pred)
        #plt.plot(cur_auc, label='%s ROC (area = %0.2f)' % (m['label'], auc))
    # Custom settings for the plot
    # confidence intervals
    tprs = np.array(tprs)
    mean_tprs = tprs.mean(axis=0)
    std = tprs.std(axis=0)
    tprs_upper = np.minimum(mean_tprs + std, 1)
    tprs_lower = mean_tprs - std
    
    r_cur_mean = data_from_R['mean.'+ str(k)]
    r_cur_low = data_from_R['lower.'+ str(k)]
    r_cur_upp = data_from_R['upper.'+ str(k)]
    # export 
    #y_pred_df.to_csv(dataset_path, encoding='utf-8')
    # ploooooooooot
    # pyhton
    plt.sca(ax)
    plt.plot(base_fpr, mean_tprs, 'b', label='Python ROC (area = %0.2f)' % (np.mean(all_auc)))
    plt.fill_between(base_fpr, tprs_lower, tprs_upper, color='#069AF3', alpha=0.3)
    # R
    plt.plot(base_fpr, r_cur_mean, 'b', label='R ROC (area = %0.2f)' % (np.mean(r_cur_mean)), color = 'red')
    plt.fill_between(base_fpr, r_cur_low, r_cur_upp, color='salmon', alpha=0.3)
    # rest
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([-0.01, 1.01])
    plt.ylim([-0.01, 1.01])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.legend(loc = 'lower right', fontsize = 15)
    

    
def plot_roc_and_ci_one_model(k, ax, data_from_py, data_from_R):
    
    """
    Takes the dataset and performs the binary classification N times on the target column, 
    then returns the AUC scores together with confidence intervals and the plot of the ROC.

    Parameters
    ----------
    N : integer
        The number of iterations under which classification is performed

    k: integer
        Index of the current plot

    ax: integer
        index of the current axis

    iris: data frame
        dataset on which classification is performed

    my_models: dictionary
        list of the models for implementing the classification

    Returns
    -------
    
    """

    # Custom settings for the plot
    # confidence intervals
    
    base_fpr = np.linspace(0, 1, 101)
    
    plt.sca(ax)
    
    py_cur_mean = data_from_py['mean.'+ str(k)]
    py_cur_low = data_from_py['lower.'+ str(k)]
    py_cur_upp = data_from_py['upper.'+ str(k)]    
    # R 
    r_cur_mean = data_from_R['mean.'+ str(k)]
    r_cur_low = data_from_R['lower.'+ str(k)]
    r_cur_upp = data_from_R['upper.'+ str(k)]
    # ploooooooooot
    # pyhton
    plt.plot(base_fpr, py_cur_mean, 'b', label='Python ROC (area = %0.2f)' % (np.mean(py_cur_mean)))
    plt.fill_between(base_fpr, py_cur_low, py_cur_upp, color='#069AF3', alpha=0.3)
    # R
    plt.plot(base_fpr, r_cur_mean, 'b', label='R ROC (area = %0.2f)' % (np.mean(r_cur_mean)), color = 'red')
    plt.fill_between(base_fpr, r_cur_low, r_cur_upp, color='salmon', alpha=0.3)
    # custom settings
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([-0.01, 1.01])
    plt.ylim([-0.01, 1.01])
    plt.ylabel('True Positive Rate', fontsize = 18)
    plt.xlabel('False Positive Rate', fontsize = 18)
    plt.legend(loc = 'lower right', fontsize = 18)


"""
A method for plotting the ROC for each of the model in the given list. 
"""    
def plot_for_every_model(data_from_py, data_from_R, my_models):
#def plot_for_every_model(N, iris, my_models, data_from_R):
    """
    Calls the get_roc_and_ci_funtcion in nested loops 

    Parameters
    ----------
    N : integer
        The number of iterations under which classification is performed

    iris: data frame
        dataset on which classification is performed

    dataset_name: string
        name of the dataset

    my_models: dictionary
        list of the models for implementing the classification
    
    Returns
    -------
    The figure grid with ROC plots for every model in a given list
    
    """
    fig, axes = plt.subplots(nrows = 3, ncols= 3)
    k = 0
    for i in range(len(axes)):
        for j in range(len(axes[0])):
            if k == len(my_models):
                break
            plot_roc_and_ci_one_model(k, axes[i][j], data_from_py, data_from_R)
            #get_roc_and_ci(N, k, axes[i][j], iris, my_models, data_from_R)
            #axes[i][j].title.set_text(my_models[k]['label'], fontsize = 10)
            axes[i][j].set_title(my_models[k]['label'], fontsize = 18)
            # next               
            k = k + 1
    fig.subplots_adjust(wspace=0.3, hspace= 0.5)
    #fig.suptitle('ROC score for tested models for %s' % dataset_name, fontsize = 25)
    fig.suptitle('')
    plt.show()
    
    #print('all plots should be printed')


