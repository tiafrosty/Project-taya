import pandas as pd

import numpy as np

from sklearn import datasets
import sklearn.linear_model as sk
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
from matplotlib import pyplot as plt
# Import modules from my package
from my_preprocessing import cancer_preproc
from plotting import get_roc_and_ci, plot_for_every_model, make_plots_compared, plot_roc_all_models_one_plot, plot_roc_all_models_one_plot_data 
from get_auc import get_auc_for_every_model


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

def make_categorical(dataset, features):
        for cur_feature in features:
                dataset[cur_feature] =  lab_enc.fit_transform(dataset[cur_feature].astype('category'))
                
        return(dataset)

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

####### blood cancer
blood_cancer_data = pd.DataFrame(pd.read_csv("blood_cancer.csv").dropna().drop_duplicates()).drop(['Unnamed: 0'], axis = 1)
# change the type of categorical features
blood_cancer_data = make_categorical(blood_cancer_data, ['target', 'Breed', 'Gender'])

################### Import and pre-proccessing is done ########

cv_score = 5

# models and parameter tuning
my_models = [

{
        'label': 'Logistic Regression',
        'model': sk.LogisticRegression(),
        'grid_params': None
 },
 {
        'label': 'Elastic net',
        'model': sk.LogisticRegression(penalty= 'elasticnet', solver = 'saga', l1_ratio = 0.5), # , solver = "newton-cg" #SGDClassifier(loss='log_loss', penalty='elasticnet', max_iter=100000),
        'grid_params': { 'l1_ratio': np.array([0.4, 0.5, 0.7, 0.9])}
},
{
        'label': 'Linear Discriminant Analysis',
        'model': LinearDiscriminantAnalysis(),
        'grid_params': None
},
{
        'label': 'KNN',
        'model': KNN(),
        'grid_params':  {'n_neighbors' : [5, 7, 9, 11, 13]} #, 'weights' : ['uniform','distance']}
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
        'grid_params': {'n_estimators': [50, 100, 200], 'max_depth': [5, 10]}
},

{
        # C is the penalty, gamma measures how far away can influencing points be
        'label': 'Linear SVM',
        'model': SVC(kernel= 'linear', C = 1, probability=True),
        'grid_params': None  # ?
},
{
        'label': 'Non-linear SVM',
        'model': SVC(kernel='rbf', probability=True), # can also try poly/sigmoid/etc, rbf is a default one
        # C penalty term, gamma is hwo far the points considered from the hyperplane
        # high gamma: only the close points are considered, low gamma: far away points are considered
        'grid_params': {'C': np.logspace(-2, 5, 4), 'gamma':  np.logspace(-9, 7, 4)}
}
]

model_labels_ptb = ['Log Reg', 'Elastic net', 'KNN', 'RF', 'Lin SVM', 'Non-linear SVM']

model_labels_rest = ['Log Reg', 'Elastic net', 'LDA', 'KNN', 'DT', 'RF', 'Lin SVM', 'Non-linear SVM']


my_models_ptb = ['Logistic Regression', 'Elastic net', 'KNN', 'RF', 'Linear SVM', 'Non-linear SVM']
# the plots to show ROC curves and CI intervals
ci_data_R_iris = pd.DataFrame(pd.read_csv("ci_data_R_iris.csv"))
ci_data_R_prostate = pd.DataFrame(pd.read_csv("ci_data_R_prostate.csv"))
ci_data_R_heart = pd.DataFrame(pd.read_csv("ci_data_R_heart.csv"))
ci_data_R_breast = pd.DataFrame(pd.read_csv("ci_data_R_breast.csv"))

ci_data_py_ptb = pd.DataFrame(pd.read_csv("keep_ci_data.csv")).drop(['Unnamed: 0'], axis = 1)

ptb_prediction_data = pd.DataFrame(pd.read_csv("keep_scores.csv")).drop(['Unnamed: 0'], axis = 1)


#roc_matrix_prostate_py = pd.DataFrame(get_auc_for_every_model(10, my_data_prostate, True, 'prostate', 'classification', 
#                                                              cv_score, my_models, ci_data_R_prostate, 0, model_labels_rest, 
#                                                              True, 500)).T


roc_matrix_iris_py = pd.DataFrame(get_auc_for_every_model(10, iris, True, 'iris', 'classification', cv_score, my_models, ci_data_R_iris,
                                                       0, model_labels_rest, True, 500)).T

roc_matrix_iris_py.boxplot()
plt.show()

#roc_matrix_iris_py.to_csv('iris_new_tuned_python.csv', encoding='utf-8')


#heart disease

# breast cancer
#roc_matrix_breast_py = pd.DataFrame(get_auc_for_every_model(1000, my_data_breast, True, 'breast','classification', cv_score, my_models, 
#                                                            ci_data_R_prostate, 0, my_models_ptb, False)).T

#roc_matrix_breast_py.to_csv('breast_new_tuned_python.csv', encoding='utf-8')



#roc_matrix_heart_py = pd.DataFrame(get_auc_for_every_model(1000, my_data_heart.head(10000), False, 'heart','classification', cv_score, my_models,
#                                                            ci_data_R_prostate, 0, my_models_ptb, False)).T

#roc_matrix_heart_py.to_csv('heart_new_tuned_python.csv', encoding='utf-8')

roc_matrix_heart_py = pd.DataFrame(pd.read_csv("heart_new_tuned_python.csv")).drop(['Unnamed: 0'], axis = 1)
roc_matrix_heart_R = pd.DataFrame(pd.read_csv("heart_new_tuned_R.csv"))
make_plots_compared(roc_matrix_heart_py, roc_matrix_heart_R, "lower right", '', models_labels)


# breast cancer
roc_matrix_breast_py = pd.DataFrame(pd.read_csv('breast_new_tuned_python.csv')).drop(['Unnamed: 0'], axis = 1)
roc_matrix_breast_R = pd.DataFrame(pd.read_csv("breast_new_tuned_R.csv"))
make_plots_compared(roc_matrix_breast_py, roc_matrix_breast_R, "lower right", '', models_labels)



roc_matrix_iris_py = pd.DataFrame(pd.read_csv('iris_new_tuned_python.csv')).drop(['Unnamed: 0'], axis = 1)
roc_matrix_iris_R = pd.DataFrame(pd.read_csv("iris_new_tuned_R.csv"))
make_plots_compared(roc_matrix_iris_py, roc_matrix_iris_R, "lower right", '', models_labels)

roc_matrix_prostate_py = pd.DataFrame(pd.read_csv('prostate_new_tuned_python.csv')).drop(['Unnamed: 0'], axis = 1)
roc_matrix_prostate_R = pd.DataFrame(pd.read_csv("prostate_new_tuned_R.csv"))
make_plots_compared(roc_matrix_prostate_py, roc_matrix_prostate_R, "lower right", '', models_labels)


# REPEAT SAME DATA
# iris disease dataset

# prostate cancer
roc_matrix_prostate_py = pd.DataFrame(get_auc_for_every_model(1000, my_data_prostate, True, 'prostate', 'classification', 
                                                              cv_score, my_models, ci_data_R_prostate, 0, my_models_ptb)).T
roc_matrix_prostate_py.to_csv('prostate_new_tuned_python.csv', encoding='utf-8')

# plot
#roc_matrix_iris_py = pd.DataFrame(pd.read_csv('iris_new_tuned_python.csv')).drop(['Unnamed: 0'], axis = 1)




#
#roc_matrix_breast_py = pd.DataFrame(pd.read_csv('breast_new_tuned_python.csv')).drop(['Unnamed: 0'], axis = 1)






# NEW blood cancer dataset
roc_matrix_blood_py = pd.DataFrame(get_auc_for_every_model(100, blood_cancer_data, False, 'iris',
                                                           'classification', cv_score, my_models, ci_data_R_iris, 0, my_models_ptb)).T
roc_matrix_blood_py.to_csv('blood_new_tuned_python.csv', encoding='utf-8')
roc_matrix_blood_read = pd.DataFrame(pd.read_csv("blood_new_tuned_python.csv")).drop(['Unnamed: 0'], axis = 1)
roc_matrix_blood_read.columns = my_models_ptb
roc_matrix_blood_read.boxplot()
plt.xlabel("", fontsize=10)
plt.ylabel("AUC", fontsize=25)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.title('AUC scores for blood cancer dataset', fontsize = 30)
plt.show()


# ROC for all models one plot
roc_matrix_blood_read = pd.DataFrame(pd.read_csv("blood_new_tuned_python.csv")).drop(['Unnamed: 0'], axis = 1)
plot_roc_all_models_one_plot(blood_cancer_data, my_models, 'blood cancer dataset')



#get_ci_blood = pd.DataFrame(get_auc_for_every_model(10, iris, False, 'blood_cancer', 'classification',
#                                                    cv_score, my_models, ci_data_R_iris, 0)).T
get_ci_blood = pd.DataFrame(get_auc_for_every_model(100, blood_cancer_data, False, 'blood_cancer', 'classification',
                                                    cv_score, my_models, ci_data_R_iris, 0, my_models_ptb)).T

plot_roc_all_models_one_plot(blood_cancer_data, my_models, 'blood cancer dataset')


plot_for_every_model(ci_data_py_ptb, ci_data_py_ptb, my_models_ptb)

plot_roc_all_models_one_plot_data(my_data_prostate, my_models_ptb, 'PTB dataset', ptb_prediction_data)

plot_roc_all_models_one_plot(my_data_prostate, my_models, 'blood cancer dataset')

#roc_matrix_iris_py = pd.DataFrame(get_auc_for_every_model(100, iris, False, 'iris', 'classification', cv_score, my_models, ci_data_R_iris)).T

# all models but SVM
roc_matrix_ptb_all = pd.DataFrame(pd.read_csv("roc_matrix_ptb_python.csv"))
# SVM
roc_matrix_ptb_svm = pd.DataFrame(pd.read_csv("roc_matrix_ptb_python_SVM.csv"))
# merge
roc_matrix_ptb = pd.concat([roc_matrix_ptb_all.reset_index(drop=True), roc_matrix_ptb_svm], axis=1).drop(['Unnamed: 0'], axis = 1)
roc_matrix_ptb.columns = model_labels_ptb
## plot
roc_matrix_ptb.boxplot()
plt.xlabel("", fontsize=20)
plt.ylabel("AUC", fontsize=20)
plt.title('AUC scores for PTB dataset', fontsize = 30)
plt.xticks(fontsize=20)
plt.yticks(fontsize=25)
plt.show()




roc_matrix_iris_py = pd.DataFrame(get_auc_for_every_model(100, my_data_heart.head(1000), False, 
                                                          'heart', 'classification', cv_score, my_models, ci_data_R_heart)).T

roc_matrix_iris_py = pd.DataFrame(get_auc_for_every_model(100, my_data_breast, True, 
                                                          'breast', 'classification', cv_score, my_models, ci_data_R_breast)).T

plot_for_every_model(100, iris, my_models, ci_data_R_iris)

#plot_for_every_model(1000, my_data_heart.head(1000), 'heart disease', my_models, ci_data_R)
#plot_for_every_model(10, my_data_prostate, 'prostate cancer', my_models, ci_data_R)
plot_for_every_model(100, my_data_breast, my_models, ci_data_R_breast)

plot_for_every_model(100, my_data_prostate, 'prostate cancer', my_models, ci_data_R_prostate)
plot_for_every_model(100, my_data_heart.head(1000), 'heart disease', my_models, ci_data_R_heart)


# get the results from R
roc_matrix_iris_R = pd.DataFrame(pd.read_csv("iris_new_tuned_R.csv"))
make_plots_compared(roc_matrix_iris_py, roc_matrix_iris_R, "lower right", '', models_labels)

#roc_matrix_iris_py.to_csv('iris_new_tuned_python.csv', encoding='utf-8')
# prostate cancer
roc_matrix_prostate_py = pd.DataFrame(get_auc_for_every_model(10, my_data_prostate, False, 'prostate', 'classification', cv_score, my_models)).T
roc_matrix_prostate_R = pd.DataFrame(pd.read_csv("prostate_new_tuned_R.csv"))
make_plots_compared(roc_matrix_prostate_py, roc_matrix_prostate_R, "lower right", '', models_labels)
#roc_matrix_prostate_py.to_csv('prostate_new_tuned_python.csv', encoding='utf-8')


# heart disease 
#
# breast cancer
roc_matrix_breast_py = pd.DataFrame(get_auc_for_every_model(10, my_data_breast, False, 'breast', 'classification', cv_score, my_models)).T
roc_matrix_breast_R = pd.DataFrame(pd.read_csv("breast_new_tuned_R.csv"))
make_plots_compared(roc_matrix_breast_py, roc_matrix_breast_R, "lower right", '', models_labels)
#roc_matrix_breast_py.to_csv('breast_new_tuned_python.csv', encoding='utf-8')
