import numpy as np
import pandas as pd
from time import time
from matplotlib import pyplot as plt
from sklearn import preprocessing
import seaborn as sns
from sklearn.pipeline import make_pipeline
# for CV
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.metrics import roc_auc_score, roc_curve
from tqdm import tqdm
from plotting import plot_for_every_model
"""
A method for computing the table of the AUC scores given amount of times for given list of models for given dataset. 
""" 
def get_auc_for_every_model(N, iris, scale, dataset_name, task, cv_score, my_models, data_from_R):
    
    """
    Takes the dataset and performs the binary classification of the target column using each of the models in a given list 
    for a given number of iterations. Returns the computed AUC scores as a data frame.

    Parameters
    ----------
    N : integer
        The number of iterations 
    iris: data frame
        The dataset for which to perform binary classification
    scale: boolean
        Indicates wether or not features need to be scaled
    dataset_name: string
        Name of the dataset
    task: string
        Indicates which type of problem needs to be solved (classification or regression). Currently not used but is planned to use later.    
    cv_score: integer
        The value of k in k-fold cross-validation performed for grid search
    my_models: dictionary
        The list of models used for implementing the binary classification
    
    Returns
    -------
    The table  of size Nxk of obtained AUC scores, where N is the number of iterations and k is the number of models 

    """

    # Prepare the data
    if task == 'classification':
        y = iris['target'].astype('category')
    else:
        y = iris['target']
    X = iris.drop(['target'], axis=1)

    #if dataset_name == 'ptb':
    #    for cur_col in X.columns:
    #        X[cur_col] = lab_enc.fit_transform(X[cur_col].astype('category'))
    # Splitting into train and test
    # take 70% for training
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3, random_state=1)

    kfold = KFold(cv_score, shuffle=True)#, random_state = i)

    # auc = np.mean(roc_scores)

    # metrics for all models:
    all_rocs = []
    aucs_best = []
    # to keep the  times
    all_times = []

    # data for plotting ROC and CI
    all_ci_data_df = pd.DataFrame()
    
    # a counter to assign a correct name to a df column, i.e. mean.0, mean.1 and so on
    k = 0
    
    for m in my_models:

        model = m['model']  # select the model

        print('\n', m['label'])
        #if m['label'] == 'KNN':
        # aa = 1

        # for models with parameters grid
        params = m['grid_params']

        scaler = preprocessing.MinMaxScaler()
        if params:
            if scale:
                gs = GridSearchCV(model, params, cv=kfold, refit=True, scoring='roc_auc', verbose=1)
                model_scaled = make_pipeline(scaler, gs)
                # choose the best model
                best_par = model_scaled.fit(X_train, y_train)[1].best_params_
                model.set_params(**best_par)
            else:
                gs = GridSearchCV(model, params, cv=kfold, refit=True, scoring='roc_auc', verbose=1)
                best_par = gs.fit(X_train, y_train).best_params_
                model.set_params(**best_par)
        if scale:
            model = make_pipeline(scaler, model)


        # THIS IS FOR THE
        # create new splits N times and fit the best model
        all_roc_scores = []
        # all values for true positive 
        tprs = []
        # all auc scores
        all_auc = []
        # check time
        t = time()
        base_fpr = np.linspace(0, 1, 101)
        #for i in range(N):
        
        for i in tqdm(range(N)):
            # make a new split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=i)
            # fit the bext model
            model.fit(X_train, y_train)
            y_pred = model.predict_proba(X_test)[:, 1]
            all_roc_scores.append(roc_auc_score(y_test, y_pred))
            ###### 
            fpr, tpr, thresholds = roc_curve(y_test, y_pred)
            tpr = np.interp(base_fpr, fpr, tpr)
            tpr[0] = 0.0
            # py
            tprs.append(tpr)

        # somewhere here I need to plot
        # keep the values: mean, lower, upper, aucs
        tprs = np.array(tprs)
        mean_tprs = tprs.mean(axis=0)
        std = tprs.std(axis=0)
        tprs_upper = np.minimum(mean_tprs + std, 1)
        tprs_lower = mean_tprs - std 
        
        #all_ci_data_df['mean.'+str(k)] = mean_tprs
        cur_column = {'mean.'+str(k): mean_tprs, 'upper.'+str(k): tprs_upper, 'lower.'+str(k): tprs_lower}
        all_ci_data_df = all_ci_data_df.assign(**cur_column) 
        # check times
        all_times.append(round(time() - t, 2))
        all_rocs.append(all_roc_scores)

        #print(f'\n Model {m["label"]} took {time() - t:.2f}s')
        
        k = k + 1


        print(f'\n Model {m["label"]} returned average AUC {np.mean(all_roc_scores)}')



    print(all_times)
    
    # PLOT
    plot_for_every_model(all_ci_data_df, data_from_R, my_models)
    #all_ci_data_df.columns = ['mean.'+str(k), 'lower.'+str(k), 'upper.'+str(k)]

    return all_rocs
