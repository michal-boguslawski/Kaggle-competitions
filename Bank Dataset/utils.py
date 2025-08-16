import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, classification_report


months_order = [
    "jan",
    "feb",
    "mar",
    "apr",
    "may",
    "jun",
    "jul",
    "aug",
    "sep",
    "oct",
    "nov",
    "dec",
]

def convert_months_to_categorical(df: pd.DataFrame) -> pd.DataFrame:
    months_order = [
        'jan', 'feb', 'mar', 'apr', 'may', 'jun', 
        'jul', 'aug', 'sep', 'oct', 'nov', 'dec'
    ]
    df = df.copy()
    
    # Convert month column to categorical with ordering if it exists and is a string/object dtype
    if "month" in df.columns:
        if df["month"].dtype == "object" or pd.api.types.is_string_dtype(df["month"]):
            df["month"] = df["month"].str.replace('-', '_')
        df["month"] = pd.Categorical(df["month"], categories=months_order, ordered=True)

    # For all object columns, only apply .str.replace if dtype is object and contains strings
    for col in df.select_dtypes(include='object').columns:
        # Make sure column is string-like before applying .str
        if pd.api.types.is_string_dtype(df[col]):
            df[col] = df[col].str.replace('-', '_')
    
    df["balance_by_duration"] = df["balance"] / df["duration"]
    df["campaign_by_duration"] = df["campaign"] / df["duration"]
    df["pdays_by_duration"] = df["pdays"] / df["duration"]
    df["previous_by_duration"] = df["previous"] / df["duration"]
    
    df["log_duration"] = np.log(df["duration"])

    return df



def calculate_score(y: pd.Series | np.ndarray, pred_probs: pd.Series | np.ndarray, treshold: float = 0.5):
    preds = pred_probs > treshold
    classification_report(y, preds)
    return {
        "auc": roc_auc_score(y, pred_probs),
        "accuracy": accuracy_score(y, preds),
        "f1": f1_score(y, preds)
    }

from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np

class ClipValues(BaseEstimator, TransformerMixin):
    def __init__(self, lower=None, upper=None, quantile_range=None):
        """
        Parameters
        ----------
        lower : float or None
            Lower bound to clip to (if not using quantiles)
        upper : float or None
            Upper bound to clip to (if not using quantiles)
        quantile_range : tuple(float, float) or None
            If provided, will compute these quantiles per column 
            and use them as lower/upper bounds. Example: (0.01, 0.99)
        """
        self.lower = lower
        self.upper = upper
        self.quantile_range = quantile_range
        self.bounds_ = {}

    def fit(self, X, y=None):
        X_df = pd.DataFrame(X)
        
        if self.quantile_range is not None:
            q_low, q_high = self.quantile_range
            for col in X_df.columns:
                q1 = X_df[col].quantile(q_low)
                q3 = X_df[col].quantile(q_high)
                iqr = q3 - q1
                self.bounds_[col] = (
                    max(q1 - 1.5 * iqr, X_df[col].min()),  # Lower bound should not go below actual min
                    min(q3 + 1.5 * iqr, X_df[col].max())   # Upper bound should not go above actual max
                )

        else:
            self.bounds_ = {col: (self.lower, self.upper) for col in X_df.columns}
        
        return self

    def transform(self, X):
        X_df = pd.DataFrame(X).copy()
        for col, (low, high) in self.bounds_.items():
            X_df[col] = X_df[col].clip(lower=low, upper=high)
        return X_df
    
class SinePreprocess(BaseEstimator, TransformerMixin):
    def __init__(self, max_vals: dict):
        super().__init__()
        self.max_vals = max_vals
        
    def fit(self, X=None, y=None):
        return self
        
    def transform(self, X):
        X_df = pd.DataFrame(X).copy()
        features_names_out = []
        for col in X.columns:
            if pd.api.types.is_categorical_dtype(X_df[col]):
                if pd.api.types.is_categorical_dtype(X_df[col]) or X_df[col].dtype == object:
                    X_df[col] = X_df[col].cat.codes if pd.api.types.is_categorical_dtype(X_df[col]) else X_df[col].astype("category").cat.codes

                # codes = X_df[col].cat.codes
                # Sine and cosine on codes normalized by max code + 1
                # max_code = codes.max() + 1
                X_df[f"{col}_sin"] = np.sin(2 * np.pi * X_df[col] / self.max_vals[col])
                X_df[f"{col}_cos"] = np.cos(2 * np.pi * X_df[col] / self.max_vals[col])
            else:
                X_df[f"{col}_sin"] = np.sin(2 * np.pi * X_df[col] / self.max_vals[col])
                X_df[f"{col}_cos"] = np.cos(2 * np.pi * X_df[col] / self.max_vals[col])
                features_names_out.append(f"{col}_sin")
                features_names_out.append(f"{col}_cos")
        return X_df.loc[:, features_names_out]


class CategoryCounter(BaseEstimator, TransformerMixin):
    def __init__(self, columns=None):
        self.columns = columns

    def fit(self, X, y=None):
        self.category_counts_ = {}
        for col in self.columns:
            self.category_counts_[col] = X[col].value_counts()
        return self

    def transform(self, X):
        X = X.copy()
        for col in self.columns:
            counts = self.category_counts_[col]
            X[f"{col}_count"] = X[col].map(counts).fillna(0).astype(int)
        return X
