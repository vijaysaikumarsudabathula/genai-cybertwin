"""
preprocessing.py
Handles cleaning that must be *fit on training data only* to avoid leakage:
  - winsorization bounds for structurally non-negative / extreme-range features
Everything here is a sklearn-compatible transformer so it can be pickled
alongside the model as a single preprocessing pipeline.
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

WINSORIZE_COLS = ["Payload_Entropy", "Login_Attempts", "Session_Duration",
                   "Packet_Retry_Rate", "Device_Response_Time"]
LOWER_Q, UPPER_Q = 0.01, 0.99


class Winsorizer(BaseEstimator, TransformerMixin):
    """Clips a fixed set of columns to percentile bounds learned on the
    training data only. Leaves columns not present in `columns` untouched."""

    def __init__(self, columns=None, lower_q=LOWER_Q, upper_q=UPPER_Q):
        self.columns = columns or WINSORIZE_COLS
        self.lower_q = lower_q
        self.upper_q = upper_q

    def fit(self, X: pd.DataFrame, y=None):
        self.bounds_ = {}
        for c in self.columns:
            if c in X.columns:
                self.bounds_[c] = (X[c].quantile(self.lower_q), X[c].quantile(self.upper_q))
        return self

    def transform(self, X: pd.DataFrame):
        X = X.copy()
        for c, (lo, hi) in self.bounds_.items():
            if c in X.columns:
                X[c] = X[c].clip(lo, hi)
        return X
