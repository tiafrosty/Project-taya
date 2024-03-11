import pandas as pd

import numpy as np

from time import time
from sklearn import datasets
import sklearn.linear_model as sk
from sklearn.svm import SVC
#from matplotlib import pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn import preprocessing
#import seaborn as sns
from sklearn.pipeline import make_pipeline,  Pipeline
# for CV
from sklearn.model_selection import train_test_split,  cross_val_score, KFold, GridSearchCV
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
# this is from my package
from my_preprocessing import cancer_preproc
from plotting import get_all_roc

models_labels = ['Log Reg', 'Elastic net',
                 'LDA', 'KNN', 'DT', 'RF', 'Linear SVM', 'Non-linear SVM']

# for categorical features
lab_enc = preprocessing.LabelEncoder()

############# IRIS ##################################

iris = datasets.load_iris() #Loading the dataset
f_names = iris.feature_names
iris = pd.DataFrame(
    data=np.c_[iris['data'], iris['target']],
    columns=iris['feature_names'] + ['target']
)

# CLASSIFICATION
pre_term_data = pd.DataFrame(pd.read_csv("whole_cleaned_data_RAB_2019DEC23.csv").dropna().drop_duplicates()).rename(columns={'out': 'target'})
pre_term_data_nuli = pre_term_data.loc[pre_term_data['parity_cat'] == 'Nuliparous']
ptb_new_features = pre_term_data_nuli[[ 'target', 'matage_cat', 'pre.ext.h_cat', 'gender',
       'prev_abortion_cat',
       'infection_cat', 'height_cat',
       'alcohl_cat', 'diabetes_cat', 'drug_cat',
       'hyper_cat', 'breastfeed_cat', 'BMI.cat', 'health_con_cat', 'smok_ft',
       'folik_cat', 'incom_cat', 'educ_cat', 'minorit_cat',
       'imq_cat', 'resid_smoker_cat', 'conception_cat',
       'medical_exp_cat', 'PAPP_mom_cat',
       'WG_FT1']]

ptb_new_features['target'] =  lab_enc.fit_transform(ptb_new_features['target'].astype('category'))
#ptb_new_features['matage_cat'] =  lab_enc.fit_transform(ptb_new_features['matage_cat'].astype('category'))
#ptb_new_features =  lab_enc.fit_transform(ptb_new_features.astype('category'))

# make binary for now
iris.loc[iris["target"] != 1,  "target"] = 0
###############################################
############### BREAST CANCER ##################
my_data_breast = cancer_preproc("breast-cancer.csv")

############## PROSTATE CANCER #######################
my_data_prostate = cancer_preproc("Prostate_Cancer.csv")

################3 HEART DISEASE ###########################
my_data_heart = pd.DataFrame(pd.read_csv("heart_disease.csv").dropna().drop_duplicates()).rename(columns={'HeartDiseaseorAttack': 'target'})
my_data_heart['target'] =  lab_enc.fit_transform(my_data_heart['target'].astype('category'))
###########################################################
################### Import and pre-proccessing is done ########

cv_score = 5

# models and parameter tuning
my_models = [

{
        'label': 'Logistic Regression',
        'model': sk.LogisticRegression(max_iter=100000), # , solver = "newton-cg"
        'grid_params': None
},
{
        'label': 'Elastic net',
        'model': SGDClassifier(loss='log_loss', penalty='elasticnet', max_iter=100000),
        'grid_params':   {'alpha': [0.1, 1, 10, 0.01], 'l1_ratio': np.array([0.4, 0.5,  0.7, 0.9])}
},
{
        'label': 'Linear Discriminant Analysis',
        'model': LinearDiscriminantAnalysis(),
        'grid_params': None
},
{
        'label': 'KNN',
        'model': KNN(),
        'grid_params':  {'n_neighbors' : [5,7,9,11,13]} #, 'weights' : ['uniform','distance']}
        #'metric' : ['minkowski','euclidean','manhattan']}
},
{
        'label': 'Decision tree',
        'model': DecisionTreeClassifier(),
        'grid_params': {'ccp_alpha' : [0, 0.2, 0.45]}
},
{
        'label': 'Random Forest',
        'model': RandomForestClassifier(max_features = 'sqrt'),
        'grid_params': {'n_estimators': [50, 100, 200]}
},

{
        # C is the penalty
        'label': 'Linear SVM',
        'model': SVC(kernel= 'linear', C = 1, probability=True),
        'grid_params': None  # ?
},
{
        'label': 'Non-linear SVM',
        'model': SVC(kernel='rbf', probability=True), # can also try poly/sigmoid/etc, rbf is a default one
        # C penalty term, gamma is hwo far the points considered from the hyperplane
        # high gamma: only the close points are considered, low gamma: far away points are considered
        'grid_params':  None # {'C': np.logspace(-2, 5, 4), 'gamma':  np.logspace(-9, 7, 4)}
}
]


# pre term birth
# roc_matrix_ptb_py = pd.DataFrame(get_all_roc(1, ptb_new_features, False, 'ptb', 'classification')).T
# roc_matrix_ptb_py.to_csv('C://Users//user//Downloads//ptb_new_tuned_python.csv', encoding='utf-8')

# iris disease dataset

# prostate cancer
roc_matrix_prostate_py = pd.DataFrame(get_all_roc(10, my_data_prostate, False, 'prostate', 'classification', cv_score, my_models)).T
roc_matrix_prostate_py.to_csv('prostate_new_tuned_python.csv', encoding='utf-8')


# WITH THE FINAL DATASETS
# iris
roc_matrix_iris_R = pd.DataFrame(pd.read_csv("C://Users//user//Downloads//iris_new_tuned_R.csv"))
roc_matrix_iris_py = pd.DataFrame(pd.read_csv("C://Users//user//Downloads//iris_new_tuned_python.csv")).drop(['Unnamed: 0'], axis = 1)
make_plots_compared(roc_matrix_iris_py, roc_matrix_iris_R, "lower right", '', models_labels)

roc_matrix_prostate_R = pd.DataFrame(pd.read_csv("C://Users//user//Downloads//cancer_new_tuned_R.csv"))
roc_matrix_prostate_py = pd.DataFrame(pd.read_csv("C://Users//user//Downloads//prostate_new_tuned_python.csv")).drop(['Unnamed: 0'], axis = 1)
make_plots_compared(roc_matrix_prostate_py, roc_matrix_prostate_R, "lower right", '', models_labels)

# heart, first 10000
roc_matrix_heart_R = pd.DataFrame(pd.read_csv("C://Users//user//Downloads//heart_new_tuned_R.csv"))
roc_matrix_heart_py = pd.DataFrame(pd.read_csv("C://Users//user//Downloads//heart_new_tuned_python.csv")).drop(['Unnamed: 0'], axis = 1)
make_plots_compared(roc_matrix_heart_py, roc_matrix_heart_R, "lower right", '')

# Read the AUC data from an already existing file
roc_matrix_breast_R = pd.DataFrame(pd.read_csv("C://Users//user//Downloads//breast_new_tuned_R.csv"))
roc_matrix_breast_py = pd.DataFrame(pd.read_csv("C://Users//user//Downloads//breast_new_tuned_python.csv")).drop(['Unnamed: 0'], axis = 1)
make_plots_compared(roc_matrix_breast_py, roc_matrix_breast_R, "lower right", '')

roc_matrix_iris_py = pd.DataFrame(get_all_roc(1000, iris, False, 'iris', 'classification')).T
roc_matrix_iris_py.to_csv('C://Users//user//Downloads//iris_new_tuned_python.csv', encoding='utf-8')

roc_matrix_heart_py = pd.DataFrame(get_all_roc(1000, my_data_heart.head(1000), False, 'heart', 'classification')).T
roc_matrix_heart_py.to_csv('C://Users//user//Downloads//heart_new_tuned_python.csv', encoding='utf-8')

# breast
roc_matrix_breast_py = pd.DataFrame(get_all_roc(1000, my_data_breast, True, 'breast', 'classification')).T
roc_matrix_breast_py.to_csv('C://Users//user//Downloads//breast_new_tuned_python.csv', encoding='utf-8')

all_aucs_df = pd.DataFrame()


roc_matrix_heart_py = pd.DataFrame(get_all_roc(1000, my_data_heart.head(1000), False, 'heart')).T
roc_matrix_heart_py.to_csv('C://Users//user//Downloads//heart_new_tuned_python.csv', encoding='utf-8')


# breast cancer dataset
cc = pd.DataFrame(get_all_roc(1000, my_data_breast, True, 'breast')).T
cc.to_csv('C://Users//user//Downloads//breast_new_tuned_python.csv', encoding='utf-8')
aa = 1


#get_all_roc(10, iris, False, 'iris')
#get_all_roc(10, my_data_prostate, False, 'prostate')
# heart disease
#roc_matrix_heart_py = pd.DataFrame(get_all_roc(1000, my_data_heart.head(1000), False, 'heart')).T
#roc_matrix_heart_py.to_csv('C://Users//user//Downloads//heart_new_tuned_python.csv', encoding='utf-8')

#roc_matrix_prostate_py_old = pd.DataFrame(pd.read_csv("C://Users//user//Downloads//prostate_new_tuned_python.csv")).drop(['Unnamed: 0'], axis = 1)
#roc_matrix_prostate_py_old.boxplot()

#make_plots_compared(roc_matrix_prostate_py, roc_matrix_prostate_R, "lower right", 'prostate cancer dataset')

roc_matrix_heart_R = pd.DataFrame(pd.read_csv("C://Users//user//Downloads//heart_new_tuned_R.csv"))
#roc_matrix_heart_R.boxplot()

# plot this shit
#figure, axis = plt.subplots(2, 2)
#make_plots_compared(roc_matrix_heart_py, roc_matrix_heart_R, "lower right", )
#make_plots_compared(roc_matrix_prostate_py, roc_matrix_prostate_R, "lower right")