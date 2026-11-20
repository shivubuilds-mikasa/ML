"""Leakage-safe preprocessing utilities."""
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from config import NUMERIC_FEATURES, CATEGORICAL_FEATURES


def build_preprocessor() -> ColumnTransformer:
    numeric = Pipeline([("imputer", SimpleImputer(strategy="median"))])
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    # Tree-based models do not require feature scaling; numeric values pass
    # through unchanged, and fit happens inside the train-only Pipeline.
    return ColumnTransformer([
        ("numeric", numeric, NUMERIC_FEATURES),
        ("categorical", categorical, CATEGORICAL_FEATURES),
    ])
