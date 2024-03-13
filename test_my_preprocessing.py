import my_preprocessing
import pandas as pd


def test_my_preprocessing():

    # original data
    raw_data = pd.DataFrame(pd.read_csv('breast-cancer.csv'))
    # preprocessed data
    new_dataset = my_preprocessing.cancer_preproc('breast-cancer.csv') 

    # test if the data has column named 'target'
    assert 'target' in new_dataset.columns

    # test that target column is binary
    assert new_dataset['target'].isin([0,1]).all()
    
    # test if id columns is removed 
    assert 'id' not in new_dataset.columns
    
    # test if processed dataset has the same number of features (-1 since we removed id columns)
    assert new_dataset.shape[1] == raw_data.shape[1]-1

