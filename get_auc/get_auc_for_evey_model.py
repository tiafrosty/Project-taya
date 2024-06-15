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
import rpy2.robjects as robjects
import random as rand

r = robjects.r
set_seed = r('set.seed')


"""
A method for creating test and train split the same as in R
"""

def get_test_train(X, y, rand_state, tr_fraction, overlap_frac):
    
    robjects.globalenv['len_X'] = len(X) - 1
    robjects.globalenv['rand_state'] = rand_state
    robjects.r("""
    set.seed(rand_state)
    inds = sample(0:len_X)
     """)

    indices = robjects.r['inds']
    #print(indices[:5])
    split_point = int(tr_fraction * len(X))

    # Split the indices
    train_indices = indices[:split_point]
    test_indices = indices[split_point:]
    # try with overlapping
    len_overlap =int(len(test_indices)*(overlap_frac/100))
    test_indices = rand.sample(list(train_indices), len_overlap) + rand.sample(list(indices[split_point:]), len(test_indices) - len_overlap)


    # Select the data based on the indices
    X_train = X.iloc[train_indices]
    X_test = X.iloc[test_indices]
    
    y_train = y.iloc[train_indices]
    y_test = y.iloc[test_indices]

    
    return X_train, X_test, y_train, y_test, train_indices, test_indices
 


"""
A method for computing the table of the AUC scores given amount of times for given list of models for given dataset. 
""" 
def get_auc_for_every_model(N, iris, scale, dataset_name, task, cv_score, my_models, data_from_R, ci_data_needed, model_labels,
                            overfit_data_needed, N_overfit):
    
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
    
    ########## try to make the split the same as in R
    

#print(np.array(data))

#data[1] = 1.0

    X_train, X_test, y_train, y_test, tr_inds, test_inds = get_test_train(X, y, 2000, 0.7, 0) #train_test_split(X,y,test_size=0.3, random_state=1) #  
    
    # remember the training set to see if later it overlaps with the test dataset
    X_train_fixed = tr_inds
    X_test_fixed = test_inds

    kfold = KFold(cv_score, shuffle=True)#, random_state = i)

    # auc = np.mean(roc_scores)

    # metrics for all models:
    all_rocs = []

    # to keep the  times
    all_times = []

    # data for plotting ROC and CI
    all_ci_data_df = pd.DataFrame()

    
    # a counter to assign a correct name to a df column, i.e. mean.0, mean.1 and so on
    k = 0
    
    # for plotting mean AUC and CI
    keep_ci_data = pd.DataFrame()
    
    all_model_auc_overfit = pd.DataFrame()
    
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
                
            print(best_par)
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
        
        all_overfit = []
       
        # check the fraction of the data from the test set that overlaps with the training data
        
        for i in tqdm(range(N)):
   
            if overfit_data_needed:
                
                cur_overfit = []
                
                for j in range(N_overfit):
                     #rand_st = rand.randint(0, 100000)
                    X_train, X_test, y_train, y_test, tr_inds, test_inds  = get_test_train(X, y, j, 0.7, i*10)     
                    model.fit(X_train, y_train)
                    y_pred = model.predict_proba(X_test)[:, 1]
                    auc_cur = roc_auc_score(y_test, y_pred)
                    cur_overfit.append(auc_cur)      
                    
                        
                all_overfit.append(np.mean(cur_overfit))

           
            else:
            
                X_train, X_test, y_train, y_test, tr_inds, test_inds  = get_test_train(X, y, i, 0.7, 0)
            
                #X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3, random_state=1)
                # check how many observations from the new test set are in the previous train set 
                #cur_overlap = len(set(test_inds) & set(X_train_fixed))/len(test_inds)
            
                model.fit(X_train, y_train)
                y_pred = model.predict_proba(X_test)[:, 1]
                auc_cur = roc_auc_score(y_test, y_pred)
                all_roc_scores.append(auc_cur)
                

                ###### 
                fpr, tpr, thresholds = roc_curve(y_test, y_pred)
                tpr = np.interp(base_fpr, fpr, tpr)
                tpr[0] = 0.0
                # py
                tprs.append(tpr)

        # somewhere here I need to plot
        # keep the values: mean, lower, upper, aucs
        #{'ratio_model_'+str(k): overlap_fraction}
        #all_ratio_data = all_ratio_data.assign(**cur_model_percent)

        
        if overfit_data_needed:
            
            cur_column = {'mean_auc'+str(k): all_overfit}
            all_model_auc_overfit = all_model_auc_overfit.assign(**cur_column)
            
        
        tprs = np.array(tprs)
        mean_tprs = tprs.mean(axis=0)
        std = tprs.std(axis=0)
        tprs_upper = np.minimum(mean_tprs + std, 1)
        tprs_lower = mean_tprs - std 
        
        #all_ci_data_df['mean.'+str(k)] = mean_tprs
        cur_column = {'mean_tpr'+str(k): mean_tprs, 'upper_tpr'+str(k): tprs_upper, 'lower_tpr'+str(k): tprs_lower}
        all_ci_data_df = all_ci_data_df.assign(**cur_column) 
        # check times
        all_times.append(round(time() - t, 2))
        all_rocs.append(all_roc_scores)

        #print(f'\n Model {m["label"]} took {time() - t:.2f}s')
        
        k = k + 1


        print(f'\n Model {m["label"]} returned average AUC {np.mean(all_roc_scores)}')
        
    
    
    if overfit_data_needed:
        
        for i in range(all_model_auc_overfit.shape[1]):        
            plt.plot(all_model_auc_overfit.iloc[:, i], label = model_labels[i])
        plt.xlabel('Percentage of overfitting',  fontsize=28)
        plt.ylabel('AUC',  fontsize=28)
        plt.xticks(np.arange(0, len(all_model_auc_overfit.iloc[:, 0])), labels = ['0%', '10%', '20%', '30%', '40%', '50%', '60%', '70%', '80%', '90%'])
        plt.tick_params(labelsize=23)
        plt.legend(loc = 'upper right', fontsize = 20)
        plt.show()
        
        return all_model_auc_overfit

    print(all_times)
        
    
    # PLOT
    
    #plot_for_every_model(all_ci_data_df, data_from_R, model_labels)
    
    
    #all_ci_data_df.columns = ['mean.'+str(k), 'lower.'+str(k), 'upper.'+str(k)]
    
    if ci_data_needed:
        all_ci_data_df.to_csv('keep_ci_data_'+dataset_name+'.csv', encoding='utf-8')

    return all_rocs
