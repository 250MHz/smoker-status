import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier


def count_missing_values(dataframe: pd.DataFrame):
    """https://machinelearningmastery.com/knn-imputation-for-missing-values-in-machine-learning/"""
    for column in dataframe.columns:
        # count number of rows with missing values
        n_miss = dataframe[column].isnull().sum()
        perc = n_miss / dataframe.shape[0] * 100
        print(f'> {column}, Missing {n_miss:d} ({perc:f}%)')


def plot_within_cluster_sum_squares(X: pd.DataFrame, feature_cols: list[str]):
    """Plots WCSS vs. number of clusters so you can use the elbow rule.
    k-means clustering is done on the `feature_cols` of `X`.

    Based on code from https://365datascience.com/tutorials/python-tutorials/pca-k-means/.
    """
    wcss = []
    for i in range(1, 10):
        _X = X[feature_cols]
        # Clustering
        kmeans = KMeans(n_clusters=i, random_state=0)
        kmeans.fit(_X)
        wcss.append(kmeans.inertia_)

    plt.figure(figsize=(10, 8))
    plt.plot(range(1, 10), wcss, marker='o', linestyle='--')
    plt.xlabel('Number of clusters')
    plt.ylabel('WCSS')
    plt.title('K-means with clustering')
    plt.show()


def try_clustering(
    X: pd.DataFrame, feature_cols: list[str], n_clusters: int, plot=False
) -> pd.DataFrame:
    """Create a copy of `X` and performs k-means clustering. Adds
    a `Cluster` feature with the number of the feature to the copy of
    `X` and returns the copy.

    `feature_cols` specifies which features should be used to cluster.

    `n_clusters` specifies the number of clusters.

    If `plot` is True, display a 2D plot using `feature_cols[0]` as the
    x-axis and `feature_cols[1]` as the y-axis.
    """
    X_copy = X.copy(deep=True)
    X_copy_only_feats = X_copy[feature_cols]

    # Clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    X_copy['Cluster'] = kmeans.fit_predict(X_copy_only_feats)
    X_copy['Cluster'] = X_copy['Cluster'].astype('category')

    if plot:
        sns.relplot(
            x=feature_cols[0], y=feature_cols[1], hue='Cluster', data=X_copy, height=6
        )
    return X_copy


def cluster_and_classify(
    X: pd.DataFrame,
    feature_cols: list[str],
    n_clusters: int,
    clf: RandomForestClassifier
    | LogisticRegression
    | DecisionTreeClassifier
    | KNeighborsClassifier,
    cv: int = 10,
    n_jobs: int = -1,
    test_size: float = 0.3,
) -> tuple[list[float], list[np.ndarray], list[np.ndarray], list[float]]:
    """Creates a copy of `X`. Then performs k-means clustering on the
    copy based on `feature_cols` and `n_clusters`.

    For each cluster:
    * Perform k-fold cross-validation where k is `cv` (can pass `n_jobs`
    to change how many cores are used) to find accuracy.
    * Split the copy into a training and testing set (size determined by
    `test_size`), fit `clf`, then find the FPR and TPR points for a ROC
    curve.

    Returns a 4-tuple with (list of accuracy scores from k-fold CV, list
    of FPR points, list of TPR points, list of AUC for each ROC curve).

    `X` should have the `smoking` label.
    """
    X = X.copy(deep=True)
    X = try_clustering(X, feature_cols, n_clusters)
    accuracy_scores = []
    fpr_list = []
    tpr_list = []
    AUC_list = []
    for i in range(n_clusters):
        X_group = X[X['Cluster'] == i]
        X_group.drop(['Cluster'], axis=1)
        y = X_group.pop('smoking')

        cv_accuracy = cross_val_score(
            clf, X_group, y, cv=cv, scoring='accuracy', n_jobs=n_jobs
        )
        accuracy_scores.append(cv_accuracy.mean())

        X_train, X_test, y_train, y_test = train_test_split(
            X_group, y, test_size=test_size, random_state=0
        )
        clf.fit(X_train, y_train)
        y_predict_prob = clf.predict_proba(X_test)
        fpr, tpr, thresholds = metrics.roc_curve(
            y_test, y_predict_prob[:, 1], pos_label=1
        )
        fpr_list.append(fpr)
        tpr_list.append(tpr)
        AUC_list.append(metrics.auc(fpr, tpr))
    return (accuracy_scores, fpr_list, tpr_list, AUC_list)


def make_mi_scores(
    X: pd.DataFrame, y: pd.Series, discrete_features: pd.Series
) -> np.ndarray:
    """https://www.kaggle.com/code/ryanholbrook/mutual-information"""
    mi_scores = mutual_info_classif(
        X, y, discrete_features=discrete_features, random_state=0
    )
    mi_scores = pd.Series(mi_scores, name='MI Scores', index=X.columns)
    mi_scores = mi_scores.sort_values(ascending=False)
    return mi_scores


def plot_mi_score(scores: pd.Series):
    """https://www.kaggle.com/code/ryanholbrook/mutual-information"""
    scores = scores.sort_values(ascending=True)
    width = np.arange(len(scores))
    ticks = list(scores.index)
    plt.barh(width, scores)
    plt.yticks(width, ticks)
    plt.title('Mutual Information Scores')


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
