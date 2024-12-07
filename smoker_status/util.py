import pandas as pd

def count_missing_values(dataframe: pd.DataFrame):
    """https://machinelearningmastery.com/knn-imputation-for-missing-values-in-machine-learning/"""
    for column in dataframe.columns:
        # count number of rows with missing values
        n_miss = dataframe[column].isnull().sum()
        perc = n_miss / dataframe.shape[0] * 100
        print(f'> {column}, Missing {n_miss:d} ({perc:f}%)')
