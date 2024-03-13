import pandas as pd
from sklearn import preprocessing


###### Import and pre-process all requiered datasets
###################################################3
def cancer_preproc(file_path):
    my_data_breast_cancer = (pd.DataFrame(pd.read_csv(file_path).drop_duplicates().dropna(axis = 'columns')).
                             rename(columns={'diagnosis': 'target'}))
    my_data_breast_cancer = my_data_breast_cancer.drop(['id'], axis=1).dropna()
    my_data_breast_cancer['target'] = preprocessing.LabelEncoder().fit_transform(my_data_breast_cancer['target'].astype('category'))
    return my_data_breast_cancer
