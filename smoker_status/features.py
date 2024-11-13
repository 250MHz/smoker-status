from pandas import concat, DataFrame
from sklearn.preprocessing import OneHotEncoder


def create_encoded_X(X: DataFrame) -> DataFrame:
    """Performs one hot encoding on X and returns a new data frame."""
    cat_feats = [
        'hearing(left)',
        'hearing(right)',
        'Urine protein',
        'dental caries',
    ]
    X_no_cat_feat = X.drop(cat_feats, axis=1)
    X_only_cat_feat = X[cat_feats]

    enc = OneHotEncoder()
    enc.fit(X_only_cat_feat)
    X_only_cat_feat_trans = DataFrame(
        data=enc.transform(X_only_cat_feat).toarray(),
        columns=[
            'hearing(left) - normal',
            'hearing(left) - abnormal',
            'hearing(right) - normal',
            'hearing(right) - abnormal',
            'Urine protein - negative',
            'Urine protein - trace',
            'Urine protein - 1+',
            'Urine protein - 2+',
            'Urine protein - 3+',
            'Urine protein - 4+',
            'dental caries - nonpresent',
            'dental caries - present',
        ],
    )
    return concat([X_no_cat_feat, X_only_cat_feat_trans], axis=1)
