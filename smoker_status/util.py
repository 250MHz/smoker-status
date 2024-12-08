import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def count_missing_values(dataframe: pd.DataFrame):
    """https://machinelearningmastery.com/knn-imputation-for-missing-values-in-machine-learning/"""
    for column in dataframe.columns:
        # count number of rows with missing values
        n_miss = dataframe[column].isnull().sum()
        perc = n_miss / dataframe.shape[0] * 100
        print(f'> {column}, Missing {n_miss:d} ({perc:f}%)')


def plot_feature_histograms(X: pd.DataFrame):
    """Taken from https://github.com/Koda98/smoker-status-prediction/blob/main/notebook.ipynb"""
    num_cols = len(X.columns)
    plt.figure(figsize=(16, num_cols * 1.5))
    for i, col in enumerate(X.columns):
        plt.subplot(num_cols // 2 + num_cols % 2, 4, i + 1)
        sns.histplot(x=col, hue='smoking', data=X, bins=50)
        plt.title(f'{col} Distribution')
        plt.tight_layout()
    plt.show()
