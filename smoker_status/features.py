import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder


def create_encoded_X(X: pd.DataFrame) -> pd.DataFrame:
    """Performs one hot encoding on X and returns a new data frame."""
    cat_feats = [
        'hearing(left)',
        'hearing(right)',
        'dental caries',
    ]
    X_no_cat_feat = X.drop(cat_feats, axis=1)
    X_only_cat_feat = X[cat_feats]

    enc = OneHotEncoder(dtype=np.int64)
    enc.fit(X_only_cat_feat)
    X_only_cat_feat_trans = pd.DataFrame(
        data=enc.transform(X_only_cat_feat).toarray(),
        columns=[
            'hearing(left) - normal',
            'hearing(left) - abnormal',
            'hearing(right) - normal',
            'hearing(right) - abnormal',
            'dental caries - nonpresent',
            'dental caries - present',
        ],
    )
    return pd.concat([X_no_cat_feat, X_only_cat_feat_trans], axis=1)
