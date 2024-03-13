import get_auc
from my_preprocessing import cancer_preproc

import numpy as np
import pandas as pd
import sklearn.linear_model as sk
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis



my_models = [
{
        'label': 'Logistic Regression',
        'model': sk.LogisticRegression(max_iter=100),
        'grid_params': None
 },
 {
        'label': 'Elastic net',
        'model': sk.LogisticRegression(max_iter = 100, penalty= 'elasticnet', solver = 'saga'),
        'grid_params':  { 'l1_ratio': np.array([0.4, 0.5,  0.7, 0.9])}
},
{
        'label': 'Linear Discriminant Analysis',
        'model': LinearDiscriminantAnalysis(),
        'grid_params': None
}
]

def test_get_auc():
 
 cancer_data = cancer_preproc('breast-cancer.csv' )

 auc_scores_cancer = pd.DataFrame(get_auc.get_auc_for_every_model(15, cancer_data, False, 'breast', 'classification', 5, my_models)).T
 
 # test if the returned array has as many columns as models
 len(auc_scores_cancer.columns) == len(my_models)

 # test if the returned values are in the right range (in [0, 1])
 auc_scores_cancer.isin([0, 1]).all()