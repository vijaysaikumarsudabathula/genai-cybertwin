# Feature Dictionary

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
