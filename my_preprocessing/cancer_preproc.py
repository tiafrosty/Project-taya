import pandas as pd
from sklearn import preprocessing


"""
A preprocessing method for cancer datasets. 
""" 
def cancer_preproc(file_path):

    """
    Takes the raw dataset and returns the proper data without NaN and non-informative entries.

    Parameters
    ----------
    file_path : string
        The path to the file with the raw data

    Returns
    -------
    The dataframe with pre-processed data
    
    """
    my_data_breast_cancer = (pd.DataFrame(pd.read_csv(file_path).drop_duplicates().dropna(axis = 'columns')).
                             rename(columns={'diagnosis': 'target'}))
    my_data_breast_cancer = my_data_breast_cancer.drop(['id'], axis=1).dropna()
    my_data_breast_cancer['target'] = preprocessing.LabelEncoder().fit_transform(my_data_breast_cancer['target'].astype('category'))
    return my_data_breast_cancer
