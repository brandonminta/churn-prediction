import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, RobustScaler
from sklearn.impute import SimpleImputer

# =========================================================
# CUSTOM TRANSFORMER:
# - Detecta columnas object
# - Si tienen 2 categorías → ordinal encoding 0/1
# - Si tienen >2 categorias → one-hot
# - Excepción: "Contract" → ordinal definido
# =========================================================
class DFColumnTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, ct, numeric_cols):
        self.ct = ct
        self.numeric_cols = numeric_cols
        self.feature_names_ = None

    def fit(self, X, y=None):
        self.ct.fit(X, y)

        num_cols = list(self.numeric_cols)

        cat_trans = self.ct.named_transformers_["cat"]

        cat_cols = []

        cat_cols.append(cat_trans.contract_col)
        cat_cols.extend(cat_trans.binary_cols)
        if cat_trans.nominal_feature_names is not None:
            cat_cols.extend(cat_trans.nominal_feature_names)

        self.feature_names_ = num_cols + cat_cols

        return self

    def transform(self, X):
        arr = self.ct.transform(X)
        return pd.DataFrame(arr, columns=self.feature_names_, index=X.index)

class DataFramePreparer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.binary_cols = []
        self.nominal_cols = []
        self.contract_col = "Contract"
        self.ordinal_encoder = None
        self.binary_encoder = None
        self.nominal_encoder = None
        self.nominal_feature_names = None

    def fit(self, X, y=None):
        X = X.copy()

        cat_cols = X.select_dtypes(include=["object"]).columns.tolist()

        if self.contract_col in cat_cols:
            cat_cols.remove(self.contract_col)

        for col in cat_cols:
            uniques = X[col].dropna().unique()
            if len(uniques) == 2:
                self.binary_cols.append(col)
            else:
                self.nominal_cols.append(col)

        self.ordinal_encoder = OrdinalEncoder(
            categories=[["Month-to-month", "One year", "Two year"]],
            handle_unknown="use_encoded_value",
            unknown_value=-1
        )
        if self.contract_col in X.columns:
            self.ordinal_encoder.fit(X[[self.contract_col]])

        if self.binary_cols:
            self.binary_encoder = OrdinalEncoder(
                handle_unknown="use_encoded_value",
                unknown_value=-1
            )
            self.binary_encoder.fit(X[self.binary_cols])

        if self.nominal_cols:
            self.nominal_encoder = OneHotEncoder(
                sparse_output=False,
                handle_unknown="ignore"
            )
            self.nominal_encoder.fit(X[self.nominal_cols])
            self.nominal_feature_names = self.nominal_encoder.get_feature_names_out(self.nominal_cols)

        return self

    def transform(self, X, y=None):
        X = X.copy()

        outputs = []

        if self.contract_col in X.columns:
            contract_t = self.ordinal_encoder.transform(X[[self.contract_col]])
            outputs.append(pd.DataFrame(contract_t, columns=[self.contract_col], index=X.index))

        if self.binary_cols:
            bin_t = self.binary_encoder.transform(X[self.binary_cols])
            outputs.append(pd.DataFrame(bin_t, columns=self.binary_cols, index=X.index))

        if self.nominal_cols:
            nom_t = self.nominal_encoder.transform(X[self.nominal_cols])
            outputs.append(pd.DataFrame(nom_t, columns=self.nominal_feature_names, index=X.index))

        return pd.concat(outputs, axis=1)
def build_preprocessing_pipeline(df):

    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", RobustScaler())
    ])

    ct = ColumnTransformer([
        ("num", numeric_pipeline, numeric_cols),
        ("cat", DataFramePreparer(), categorical_cols)
    ])

    return DFColumnTransformer(ct, numeric_cols)

