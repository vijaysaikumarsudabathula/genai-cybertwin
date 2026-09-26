# Dataset Report — KAUST Cyber Attack Detection

- **train.csv**: 10684 rows x 16 columns
- **test.csv**: 2672 rows x 15 columns
- **Target column**: `Attack Type` (only present in train.csv)
- **Timestamp range**: 2023-01-01 00:00:00 to 2023-02-25 13:10:00
- **Duplicate rows**: 0 (train), 0 (test)
- **Duplicate IDs**: 0 (train), 0 (test)
- **Train/test ID overlap**: 2672 rows

## Missing values
Total missing cells in train.csv: **0**. No imputation strategy is required for this dataset as-is.

## Class distribution (train.csv, Attack Type)
- Class `0`: 5369 rows (50.253%)
- Class `1`: 2667 rows (24.963%)
- Class `2`: 2648 rows (24.785%)

The classes are moderately imbalanced (class 0 is roughly 2x classes 1 and 2), which is why Macro F1 (not raw accuracy) is used as the primary model-selection metric.

## Data quality issues found on actual inspection
- `Malware_Score` contains 3405 negative values (min=-40.90, max=63.86). No data dictionary was provided with this competition, so we cannot confirm whether negative values here are errors or a meaningful signed/standardized indicator (e.g. deviation from a baseline). They are kept as-is rather than silently altered.
- `User_Behavior_Score` contains 5866 negative values (min=-4.21, max=3.38). No data dictionary was provided with this competition, so we cannot confirm whether negative values here are errors or a meaningful signed/standardized indicator (e.g. deviation from a baseline). They are kept as-is rather than silently altered.
- **`Payload_Entropy`** contains 3820 negative values (min=-102.53, max=138.24). This feature is structurally non-negative (a count, duration, rate, or entropy value), so these are treated as data-quality noise and winsorized in preprocessing.
- **`Login_Attempts`** contains 2611 negative values (min=-842866918.65, max=881837206.06). This feature is structurally non-negative (a count, duration, rate, or entropy value), so these are treated as data-quality noise and winsorized in preprocessing.
- **`Session_Duration`** contains 1384 negative values (min=-35.20, max=65.09). This feature is structurally non-negative (a count, duration, rate, or entropy value), so these are treated as data-quality noise and winsorized in preprocessing.
- **`Packet_Retry_Rate`** contains 14 negative values (min=-6.68, max=44.93). This feature is structurally non-negative (a count, duration, rate, or entropy value), so these are treated as data-quality noise and winsorized in preprocessing.
- `Anomaly_Tendency` contains 4839 negative values (min=-53.66, max=58.46). No data dictionary was provided with this competition, so we cannot confirm whether negative values here are errors or a meaningful signed/standardized indicator (e.g. deviation from a baseline). They are kept as-is rather than silently altered.

- **`Login_Attempts`** has an extreme range (min=-8.43e+08, max=8.82e+08, median=1.47e+03), consistent with the dataset description's note about extreme values in this column. Values are winsorized at the 1st/99th percentile before modeling rather than dropped, to avoid discarding potentially real events.
- **`Device_Response_Time`** has an extreme range (min=0.00e+00, max=4.00e+07, median=0.00e+00), consistent with the dataset description's note about extreme values in this column. Values are winsorized at the 1st/99th percentile before modeling rather than dropped, to avoid discarding potentially real events.
