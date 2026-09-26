"""
feature_engineering.py
Adds temporal features derived from Timestamp. Behavioral/network/threat
features are used as-provided (Suspicious_Port_Activity, Malware_Score, etc.)
since the dataset does not include raw logs to derive anything richer from.
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

NETWORK_FEATURES = ["Suspicious_Port_Activity", "Traffic_Volume_Variation",
                     "Packet_Length_Anomaly", "Payload_Entropy", "Packet_Retry_Rate"]
THREAT_FEATURES = ["Malware_Score", "Threat_Level_Index", "Anomaly_Tendency"]
USER_FEATURES = ["User_Behavior_Score", "Login_Attempts", "Geo_Dispersion"]
SESSION_FEATURES = ["Device_Response_Time", "Session_Duration"]
TEMPORAL_FEATURES = ["hour", "minute", "day", "day_of_week", "is_weekend",
                      "hour_sin", "hour_cos", "time_period"]

TIME_PERIOD_MAP = {
    range(0, 6): "night",
    range(6, 12): "morning",
    range(12, 18): "afternoon",
    range(18, 24): "evening",
}


def _time_period(hour: int) -> str:
    for rng, label in TIME_PERIOD_MAP.items():
        if hour in rng:
            return label
    return "night"


class TemporalFeatures(BaseEstimator, TransformerMixin):
    """Derives calendar/cyclical features from the Timestamp column.
    Stateless (no fitting needed), but implemented as a transformer so it
    slots into the same sklearn Pipeline as the Winsorizer."""

    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame):
        X = X.copy()
        ts = pd.to_datetime(X["Timestamp"])
        X["hour"] = ts.dt.hour
        X["minute"] = ts.dt.minute
        X["day"] = ts.dt.day
        X["day_of_week"] = ts.dt.dayofweek
        X["is_weekend"] = (ts.dt.dayofweek >= 5).astype(int)
        X["hour_sin"] = np.sin(2 * np.pi * X["hour"] / 24)
        X["hour_cos"] = np.cos(2 * np.pi * X["hour"] / 24)
        X["time_period"] = X["hour"].apply(_time_period)
        return X


FEATURE_DICTIONARY_MD = """# Feature Dictionary

## Provided features (used as-is)
| Feature | Group | Purpose |
|---|---|---|
| Suspicious_Port_Activity | Network | Indicator of scanning/unusual port usage |
| Traffic_Volume_Variation | Network | Deviation in traffic volume from baseline |
| Packet_Length_Anomaly | Network | Deviation of packet size from normal distribution |
| Payload_Entropy | Network | Randomness of payload content (encryption/obfuscation signal) |
| Packet_Retry_Rate | Network | Retransmission rate, can indicate instability or attack noise |
| Malware_Score | Threat | Model/heuristic-derived malware likelihood indicator |
| Threat_Level_Index | Threat | Aggregate composite threat indicator |
| Anomaly_Tendency | Threat | General anomaly propensity score for the event |
| User_Behavior_Score | User | Deviation of user behavior from established baseline |
| Login_Attempts | User | Number of login attempts associated with the event |
| Geo_Dispersion | User | Geographic spread of access/login locations |
| Device_Response_Time | Session | Device/service response latency |
| Session_Duration | Session | Length of the session associated with the event |

## Engineered features (derived by this project)
| Feature | Purpose |
|---|---|
| hour | Hour of day (0-23) the event occurred, for temporal pattern detection |
| minute | Minute of the hour, fine-grained temporal signal |
| day | Calendar day of month |
| day_of_week | Day of week (0=Mon..6=Sun), for weekly pattern detection |
| is_weekend | Binary flag, attacks may cluster on weekends (lower staffing) |
| hour_sin / hour_cos | Cyclical encoding of hour so 23:00 and 00:00 are numerically close |
| time_period | Categorical bucket (night/morning/afternoon/evening) for interpretability |

No other engineered features were created: the dataset provides pre-aggregated
behavioral/network/threat scores rather than raw logs, so there is nothing
further to meaningfully derive from them without inventing unsupported signal.
"""

if __name__ == "__main__":
    from pathlib import Path
    out = Path(__file__).resolve().parent.parent / "data" / "reports" / "feature_dictionary.md"
    out.write_text(FEATURE_DICTIONARY_MD)
    print("wrote", out)
