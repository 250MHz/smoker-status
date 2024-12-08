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


def set_sex(row: pd.Series) -> int:
    """Returns value for sex. Value is based on a rough heuristic using
    measured height and weight values from Table 2 of this paper
    https://doi.org/10.4178/epih.e2022024.

    -2, -1: female
    0: N/A
    1, 2: male

    Use with `pandas.DataFrame.apply`. Must use axis=1 to apply to each
    row.
    """
    # Male, increase score by 1. Female, decrease score by 1. If score == 0,
    # then neither had majority, so np.nan
    score = 0
    if row['age'] >= 19 and row['age'] <= 29:
        if row['height(cm)'] <= 165:
            score -= 1
        elif row['height(cm)'] >= 170:
            score += 1
        if row['weight(kg)'] <= 55:
            score -= 1
        elif row['weight(kg)'] >= 70:
            score += 1
        else: # [60, 65]
            pass
    elif row['age'] >= 30 and row['age'] <= 39:
        if row['height(cm)'] <= 165:
            score -= 1
        elif row['height(cm)'] >= 170:
            score += 1
        if row['weight(kg)'] <= 60:
            score -= 1
        elif row['weight(kg)'] >= 75:
            score += 1
        else: # [65, 70]
            pass
    elif row['age'] >= 40 and row['age'] <= 49:
        if row['height(cm)'] <= 160:
            score -= 1
        elif row['height(cm)'] >= 170:
            score += 1
        else: # [165]
            pass
        if row['weight(kg)'] <= 60:
            score -= 1
        elif row['weight(kg)'] >= 70:
            score += 1
        else: # [65]
            pass
    elif row['age'] >= 50 and row['age'] <= 59:
        if row['height(cm)'] <= 160:
            score -= 1
        elif row['height(cm)'] >= 165:
            score += 1
        if row['weight(kg)'] <= 60:
            score -= 1
        elif row['weight(kg)'] >= 70:
            score += 1
        else: # [65]
            pass
    elif row['age'] >= 60 and row['age'] <= 69:
        if row['height(cm)'] <= 160:
            score -= 1
        elif row['height(cm)'] >= 165:
            score += 1
        if row['weight(kg)'] <= 55:
            score -= 1
        elif row['weight(kg)'] >= 70:
            score += 1
        else: # [60, 65]
            pass
    elif row['age'] >= 70:
        if row['height(cm)'] <= 155:
            score -= 1
        elif row['height(cm)'] >= 160:
            score += 1
        if row['weight(kg)'] <= 55:
            score -= 1
        elif row['weight(kg)'] >= 65:
            score += 1
        else: # [60]
            pass
    return score


def set_anemia(row: pd.Series) -> int:
    """Returns value for anemia. Anemia is based on hemoglobin levels
    from https://www.who.int/publications/i/item/9789240088542.

    0: no anemia
    1: mild anemia
    2: moderate anemmia
    3: severe anemia

    Use with `pandas.DataFrame.apply`. Must use axis=1 to apply to each
    row. Must have the `sex` feature from `set_sex`.
    """
    hb_level = row['hemoglobin']
    if row['sex'] > 0: # male
        if hb_level >= 13:
            return 0
        elif hb_level >= 11:
            return 1
        elif hb_level >= 8:
            return 2
        elif hb_level < 8:
            return 3
    else: # female
        if hb_level >= 12:
            return 0
        elif hb_level >= 11:
            return 1
        elif hb_level >= 8:
            return 2
        elif hb_level < 8:
            return 3
