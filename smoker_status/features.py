import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def setup_X(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy(deep=True)
    X['sex'] = X.apply(set_sex, axis=1)
    # Some 'sex' values are missing, impute the rest
    imputer = KNNImputer(n_neighbors=5, missing_values=0)
    X_trans = imputer.fit_transform(X[['age', 'height(cm)', 'weight(kg)', 'sex']])
    X_full_sex = pd.DataFrame(
        X_trans, columns=['age', 'height(cm)', 'weight(kg)', 'sex']
    )
    X['sex'] = X_full_sex['sex']
    X['anemia'] = X.apply(set_anemia, axis=1)
    X['HDL class'] = X.apply(set_HDL_class, axis=1)
    X['LDL class'] = X.apply(set_LDL_class, axis=1)
    X['Cholesterol class'] = X.apply(set_cholesterol_class, axis=1)
    X['blood pressure class'] = X.apply(set_blood_pressure_class, axis=1)
    add_GGT_level(X)
    X['triglyceride class'] = X.apply(set_triglyceride_class, axis=1)
    X['creatinine class'] = X.apply(set_creatinine_class, axis=1)
    X['ALT class'] = X.apply(set_ALT_class, axis=1)
    X['AST class'] = X.apply(set_AST_class, axis=1)
    add_de_ritis_level(X)
    X['FPG class'] = X.apply(set_FPG_class, axis=1)
    add_BMI(X)
    X['BMI class'] = X.apply(set_BMI_class, axis=1)
    update_blindness_zero(X)
    X = log_transform_X(X)
    X = scale_X(X)[1]
    return X


def log_transform_X(X: pd.DataFrame, feats: list[str] = None) -> pd.DataFrame:
    """Applies `numpy.log1p` to the features in `feats` on a copy of `X`
    and returns the copy of `X`.

    If `feats` is None (default), the following features will be
    transformed:
    * fasting blood sugar
    * triglyceride
    * LDL
    * serum creatinine
    * AST
    * ALT
    * AST/ALT
    * Gtp

    If `feats` is not None, then only the features you pass will be
    transformed.
    """
    if feats is None:
        feats = [
            'fasting blood sugar',
            'triglyceride',
            'LDL',
            'serum creatinine',
            'AST',
            'ALT',
            'AST/ALT',
            'Gtp',
        ]
    X_copy = X.copy(deep=True)
    for feat in feats:
        X_copy[feat] = X_copy[feat].apply(np.log1p)
    return X_copy


def scale_X(
    X: pd.DataFrame, feats: list[str] = None
) -> tuple[StandardScaler, pd.DataFrame]:
    """Creates a standard scaler, fits on `X`, and return a 2-tuple with
    the scaler and scaled copy of X.

    If `feats` is None (default), the following features will be scaled:
    * age
    * height(cm)
    * weight(kg)
    * waist(cm)
    * systolic
    * relaxation
    * fasting blood sugar
    * Cholesterol
    * triglyceride
    * HDL
    * LDL
    * hemoglobin
    * serum creatinine
    * AST
    * ALT
    * Gtp
    * AST/ALT
    * BMI

    If `feats` is not None, then only the features you pass will be
    scaled.

    Add the extra features and log transform first before scaling if
    you are using the default `feats`.
    """
    if feats is None:
        feats = [
            'age',
            'height(cm)',
            'weight(kg)',
            'waist(cm)',
            'systolic',
            'relaxation',
            'fasting blood sugar',
            'Cholesterol',
            'triglyceride',
            'HDL',
            'LDL',
            'hemoglobin',
            'serum creatinine',
            'AST',
            'ALT',
            'Gtp',
            'AST/ALT',
            'BMI',
        ]
    X = X.copy(deep=True)
    X_without_feats = X.drop(feats, axis=1)
    X_only_feats = X[feats]

    scaler = StandardScaler()
    scaler.fit(X_only_feats)
    X_only_feats_scaled = pd.DataFrame(
        data=scaler.transform(X_only_feats), columns=feats
    )
    return (scaler, pd.concat([X_without_feats, X_only_feats_scaled], axis=1))


def one_hot_encode_X(
    X: pd.DataFrame, feats: list[str] = None, encoded_cols: list[str] = None
) -> tuple[OneHotEncoder, pd.DataFrame]:
    """Creates a `OneHotEncoder`, fits on `X`, and return a 2-tuple
    with the encoder and encoded copy of `X`.

    NOTE: I don't think that these features are actually nominal.
    The presence of cavities is worse than the absence.
    And for us, sex is a heuristic value where the lower value is
    more likely to be a female, while higher value is more likely
    to be male.
    From minor experimentation, I did not get better 10-fold CV scores
    when using encoding than without.

    If `feats` is None (default), the following features will be
    encoded:
    * hearing(left)
    * hearing(right)
    * dental caries
    * sex
    If `encoded` is None (default), the encoded features will have the
    following names:
    * hearing(left) - normal
    * hearing(left) - abnormal
    * hearing(right) - normal
    * hearing(right) - abnormal
    * dental caries - nonpresent
    * dental caries - present
    * sex - M
    * sex - F

    If `feats` or `encoded` are not None, then the passed features and
    column names will be used.

    This should be used after applying `add_sex` to `X`.
    """
    if feats is None:
        feats = [
            'hearing(left)',
            'hearing(right)',
            'dental caries',
            'sex',
        ]
    if encoded_cols is None:
        encoded_cols = [
            'hearing(left) - normal',
            'hearing(left) - abnormal',
            'hearing(right) - normal',
            'hearing(right) - abnormal',
            'dental caries - nonpresent',
            'dental caries - present',
            'sex - M',
            'sex - F',
        ]

    X = X.copy(deep=True)
    X_no_feats = X.drop(feats, axis=1)
    X_only_feats = X[feats]
    X_only_feats['sex'] = X_only_feats.apply(lambda x: 0 if x['sex'] < 0 else 1, axis=1)

    enc = OneHotEncoder(dtype=np.int64)
    enc.fit(X_only_feats)
    X_only_feats_trans = pd.DataFrame(
        data=enc.transform(X_only_feats).toarray(), columns=encoded_cols
    )
    return (enc, pd.concat([X_no_feats, X_only_feats_trans], axis=1))


def create_encoded_X(X: pd.DataFrame) -> pd.DataFrame:
    """Performs one hot encoding on X and returns a new data frame.

    NOTE: This function was written before we added extra features

    NOTE: I don't think that these features are actually nominal.
    The presence of cavities is worse than the absence.
    """
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


def update_blindness_zero(df: pd.DataFrame):
    """Sets instances of `eyesight(left)` or `eyesight(right)` which
    are 9.9 to 0.

    This matches the ordinality where 0.1 (minimum) means 20/200 VA
    and 2.0 (maximum ignoring 9.9) means 20/10 VA. It doesn't make sense
    to have blindness above the best vision.
    """
    df['eyesight(left)'] = df['eyesight(left)'].replace(to_replace=9.9, value=0)
    df['eyesight(right)'] = df['eyesight(right)'].replace(to_replace=9.9, value=0)


def set_HDL_class(row: pd.Series) -> int:
    """Returns value for HDL cholesterol class based on Adult Treatment
    Panel III's classification https://doi.org/10.1001/jama.285.19.2486.

    0: low
    1: normal
    2: high

    Use with `pandas.DataFrame.apply`. Must use axis=1 to apply to each
    row.
    """
    HDL_c = row['HDL']
    if HDL_c < 40:
        return 0
    elif HDL_c < 60:
        return 1
    elif HDL_c >= 60:
        return 2


def set_LDL_class(row: pd.Series) -> int:
    """Returns value for LDL cholesterol class based on Adult Treatment
    Panel III's classification https://doi.org/10.1001/jama.285.19.2486.

    0: optimal
    1: near or above optimal
    2: borderline high
    3: high
    4: very high

    Use with `pandas.DataFrame.apply`. Must use axis=1 to apply to each
    row.
    """
    LDL_c = row['LDL']
    if LDL_c < 100:
        return 0
    elif LDL_c < 130:
        return 1
    elif LDL_c < 160:
        return 2
    elif LDL_c < 190:
        return 3
    elif LDL_c >= 190:
        return 4


def set_cholesterol_class(row: pd.Series) -> int:
    """Returns value for cholesterol class based on Adult Treatment
    Panel III's classification https://doi.org/10.1001/jama.285.19.2486.

    0: desirable
    1: borderline high
    2: high

    Use with `pandas.DataFrame.apply`. Must use axis=1 to apply to each
    row.
    """
    cholesterol = row['Cholesterol']
    if cholesterol < 200:
        return 0
    elif cholesterol < 240:
        return 1
    elif cholesterol >= 240:
        return 2


def set_blood_pressure_class(row: pd.Series) -> int:
    """Returns value for blood pressure and hypertension class based on
    2018 Korean Society of Hypertension guidelines
    https://doi.org/10.1186/s40885-019-0121-0.

    0: normal blood pressure
    1: elevated blood pressure
    2: prehypertension
    3: hypertension grade 1
    4: hypertension grade 2
    5: isolated systolic hypertension

    Use with `pandas.DataFrame.apply`. Must use axis=1 to apply to each
    row.
    """
    systolic_bp: int = row['systolic']
    diastolic_bp: int = row['relaxation']
    if systolic_bp < 120 and diastolic_bp < 80:
        return 0
    elif systolic_bp in range(120, 130) and diastolic_bp < 80:
        return 1
    elif systolic_bp in range(130, 140) or diastolic_bp in range(80, 90):
        return 2
    elif systolic_bp in range(140, 160) or diastolic_bp in range(90, 100):
        return 3
    elif systolic_bp >= 160 or diastolic_bp >= 100:
        return 4
    elif systolic_bp >= 140 and diastolic_bp < 90:
        return 5


def set_triglyceride_class(row: pd.Series) -> int:
    """Returns value for triglyceride class based on Adult Treatment
    Panel III's classification https://doi.org/10.1001/jama.285.19.2486.

    0: normal
    1: borderline-high
    2: high
    3: very high

    Use with `pandas.DataFrame.apply`. Must use axis=1 to apply to each
    row.
    """
    tg = row['triglyceride']
    if tg < 150:
        return 0
    elif tg < 200:
        return 1
    elif tg < 500:
        return 2
    elif tg >= 500:
        return 3


def set_FPG_class(row: pd.Series) -> int:
    """Returns value for fasting plasma glucose (FPG) class based on
    https://pmc.ncbi.nlm.nih.gov/articles/PMC11307112/ and
    https://www.who.int/data/gho/indicator-metadata-registry/imr-details/2380.

    0: hypoglycemia
    1: normal
    2: prediabetes
    3: possible diabetes

    Use with `pandas.DataFrame.apply`. Must use axis=1 to apply to each
    row.
    """
    FPG = row['fasting blood sugar']
    if FPG < 70:
        return 0
    elif FPG < 100:
        return 1
    elif FPG < 126:
        return 2
    elif FPG >= 126:
        return 3


def set_BMI_class(row: pd.Series) -> int:
    """Returns BMI classification based on 2020 Korean Society for the
    Study of Obesity Guidelines for the Management of Obesity in Korea
    https://doi.org/10.7570/jomes21022.


    0: underweight
    1: normal
    2: pre-obesity (overweight)
    3: class I obesity
    4: class II obesity
    5: class III obesity

    Use with `pandas.DataFrame.apply`. Must use axis=1 to apply to each
    row. Must have `BMI` feature to use (add it using `add_BMI`).
    """
    BMI = row['BMI']
    if BMI < 18.5:
        return 0
    elif BMI < 23:
        return 1
    elif BMI < 25:
        return 2
    elif BMI < 30:
        return 3
    elif BMI < 35:
        return 4
    elif BMI >= 35:
        return 5


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
        else:  # [60, 65]
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
        else:  # [65, 70]
            pass
    elif row['age'] >= 40 and row['age'] <= 49:
        if row['height(cm)'] <= 160:
            score -= 1
        elif row['height(cm)'] >= 170:
            score += 1
        else:  # [165]
            pass
        if row['weight(kg)'] <= 60:
            score -= 1
        elif row['weight(kg)'] >= 70:
            score += 1
        else:  # [65]
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
        else:  # [65]
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
        else:  # [60, 65]
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
        else:  # [60]
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
    if row['sex'] > 0:  # male
        if hb_level >= 13:
            return 0
        elif hb_level >= 11:
            return 1
        elif hb_level >= 8:
            return 2
        elif hb_level < 8:
            return 3
    else:  # female
        if hb_level >= 12:
            return 0
        elif hb_level >= 11:
            return 1
        elif hb_level >= 8:
            return 2
        elif hb_level < 8:
            return 3


def set_creatinine_class(row: pd.Series) -> int:
    """Returns value for creatinine class based on recommended limits
    from Lee J, Kim J, Park I, et al. A Study on the Appropriate Normal
    Range of Serum Creatinine Level for Koreans. Korean J Nephrol.
    2004;23(5):721-728. https://www.koreamed.org/SearchBasic.php?RID=2307256

    0: normal
    1: abnormal

    Use with `pandas.DataFrame.apply`. Must use axis=1 to apply to each
    row. Must have the `sex` feature from `set_sex`.
    """
    sCr = row['serum creatinine']
    if row['sex'] > 0:  # male
        if sCr < 1.2:
            return 0
        else:
            return 1
    else:  # female
        if sCr < 1:
            return 0
        else:
            return 1


def set_ALT_class(row: pd.Series) -> int:
    """Returns value for ALT class based on ULN conclusion from
    https://doi.org/10.1111/j.1440-1746.2012.07143.x.

    0: normal
    1: abnormal

    Use with `pandas.DataFrame.apply`. Must use axis=1 to apply to each
    row. Must have the `sex` feature from `set_sex`.
    """
    ALT = row['ALT']
    if row['sex'] > 0:  # male
        if ALT < 34:
            return 0
        else:
            return 1
    else:  # female
        if ALT < 24:
            return 0
        else:
            return 1


def set_AST_class(row: pd.Series) -> int:
    """Returns value for AST class based on ULN conclusion from
    https://doi.org/10.1111/j.1440-1746.2012.07143.x.

    0: normal
    1: abnormal

    Use with `pandas.DataFrame.apply`. Must use axis=1 to apply to each
    row. Must have the `sex` feature from `set_sex`.
    """
    AST = row['AST']
    if row['sex'] > 0:  # male
        if AST < 32:
            return 0
        else:
            return 1
    else:  # female
        if AST < 26:
            return 0
        else:
            return 1


def add_GGT_level(df: pd.DataFrame):
    """Adds the `GGT level` feature to `df`. GGT level is the quartile
    (0 to 3) that a subject's `Gtp` value is in within their sex.

    `df` must have the `sex` feature from `set_sex`.
    """
    df.loc[df['sex'] > 0, ['GGT level']] = pd.qcut(
        df[df['sex'] > 0]['Gtp'], 4, labels=False
    ).astype(int)
    df.loc[df['sex'] < 0, ['GGT level']] = pd.qcut(
        df[df['sex'] < 0]['Gtp'], 4, labels=False
    ).astype(int)
    df['GGT level'] = df['GGT level'].astype(int)


def add_de_ritis_level(df: pd.DataFrame):
    """Adds the `AST/ALT` (De Ritis ratio) feature to `df`."""
    df['AST/ALT'] = df['AST'] / df['ALT']


def add_BMI(df: pd.DataFrame):
    """Adds a `BMI` feature to `df`."""
    df['BMI'] = df['weight(kg)'] / (df['height(cm)'] / 100) ** 2
