# Complete Guide to `TSA2(CO2_concentation)-aarchi-P2.ipynb` (Time Series Analysis - Practical 2)

This document provides a complete, step-by-step breakdown of everything performed in Aarchi's Practical 2 notebook: **`SC/Time Series Analysis/Practical 1/TSA2(CO2_concentation)-aarchi-P2.ipynb`**.

---

## 📋 Table of Contents
1. [Overview & Practical Objectives](#1-overview--practical-objectives)
2. [Key Differences from Practical 1 (AirPassengers)](#2-key-differences-from-practical-1-airpassengers)
3. [Cell-by-Cell Detailed Breakdown](#3-cell-by-cell-detailed-breakdown)
   - [Phase 1: Multi-Column Preprocessing & Datetime Indexing](#phase-1-multi-column-preprocessing--datetime-indexing)
   - [Phase 2: Additive Decomposition & Mann-Kendall Trend Test](#phase-2-additive-decomposition--mann-kendall-trend-test)
   - [Phase 3: Train-Test Split (70/30)](#phase-3-train-test-split-7030)
   - [Phase 4: Exponential Smoothing Models (SES, Holt's, Holt-Winters Add vs Mul)](#phase-4-exponential-smoothing-models-ses-holts-holt-winters-add-vs-mul)
   - [Phase 5: Accuracy Comparison (MAPE)](#phase-5-accuracy-comparison-mape)
   - [Phase 6: Initial Stationarity Diagnostics (ADF & KPSS Tests)](#phase-6-initial-stationarity-diagnostics-adf--kpss-tests)
4. [Aarchi's Exact Python Code Reference](#4-aarchis-exact-python-code-reference)

---

## 1. Overview & Practical Objectives

In Practical 2 (`TSA2(CO2_concentation)-aarchi-P2.ipynb`), Aarchi performs time series modeling on the **CO2 Concentration** dataset (`CO2 Concentration.xls`).

### Main Goals:
1. **Multi-Column Datetime Construction**: Combine separate numeric `Year` (e.g., `1958`) and `Month` (e.g., `3`) columns into an ISO string `YYYY-MM` (`1958-03`) using string formatting and zero-padding (`zfill(2)`).
2. **Additive Seasonal Decomposition**: Analyze CO2 fluctuations using an **Additive Model** because seasonal amplitudes remain constant as CO2 concentration increases over time.
3. **Mann-Kendall Monotonic Trend Test**: Confirm statistically that a monotonic upward trend exists.
4. **Exponential Smoothing Forecasting**: Compare Single, Double (Holt's Linear), and Triple Exponential Smoothing (Holt-Winters Additive vs Multiplicative).
5. **MAPE Comparison**: Evaluate why Additive and Multiplicative seasonal models produce virtually identical MAPE for uniform seasonal oscillations.
6. **Stationarity Diagnostics**: Perform baseline ADF and KPSS tests on raw CO2 concentrations.

---

## 2. Key Differences from Practical 1 (AirPassengers)

| Feature / Step | Practical 1 (AirPassengers) | Practical 2 (CO2 Concentration) |
| :--- | :--- | :--- |
| **Raw Date Format** | Single string column `Month` (`1949-01`) | Two separate columns `Year` (`1958`) and `Month` (`3`) |
| **Preprocessing** | `pd.to_datetime(data['Month'])` | `df['Year'].astype(str) + '-' + df['Month'].astype(str).str.zfill(2)` |
| **Seasonal Behavior** | Expanding oscillations (Multiplicative) | Uniform / Constant height oscillations (Additive) |
| **Decomposition Model** | `model='multiplicative'` | `model='additive'` |
| **Pegel's Chart Selection** | Additive-Multiplicative | Additive-Additive (Trend='add', Seasonal='add') |
| **MAPE Result** | Multiplicative wins significantly | Additive & Multiplicative produce identical accuracy |

---

## 3. Cell-by-Cell Detailed Breakdown

### Phase 1: Multi-Column Preprocessing & Datetime Indexing (Cells 0 - 8)
- Imports `pandas`, `numpy`, `seaborn`, `matplotlib.pyplot`, `seasonal_decompose`.
- Reads `CO2 Concentration.xls`.
- Creates `year_month` string column:
  ```python
  df['year_month'] = (
      df['Year'].astype(str) + '-' + df['Month'].astype(str).str.zfill(2)
  )
```
- Converts `year_month` to `datetime` format and sets it as the DataFrame index.
- Drops raw `Year` and `Month` columns (`df.drop(columns=['Year', 'Month'], inplace=True)`).

### Phase 2: Additive Decomposition & Mann-Kendall Trend Test (Cells 9 - 13)
- Plots line plot: `sns.lineplot(df)`.
- Identifies that seasonal amplitude stays consistent, making an **Additive Model** appropriate.
- Fits `seasonal_decompose(df[['CO2 Concentration']], model='additive', period=12)`.
- Runs `mk.original_test(df['CO2 Concentration'])` to test monotonic trend ($H_0$: No monotonic trend).

### Phase 3: Train-Test Split (70/30) (Cells 14 - 15)
- Splits chronological observations:
  - `train_df = df[:int(df.shape[0]*0.7)]`
  - `test_df = df[int(df.shape[0]*0.7):]`

### Phase 4: Exponential Smoothing Models (Cells 16 - 32)
Fits four models on `train_df`:
1. **Single Exponential Smoothing (`SimpleExpSmoothing`)**:
   - `model = SimpleExpSmoothing(train_df)`
   - `model_single_fit = model.fit()`
   - Forecast: `forecast_single = model_single_fit.forecast(len(test_df))`
2. **Double Exponential Smoothing / Holt's Model (`Holt`)**:
   - `model_double = Holt(train_df)`
   - `model_double_fit = model_double.fit()`
   - Forecast: `forecast_double = model_double_fit.forecast(len(test_df))`
3. **Triple Exponential Smoothing Additive (`ExponentialSmoothing`)**:
   - Based on Pegel's chart, sets `trend="add"`, `seasonal="add"`, `seasonal_periods=12`.
   - `model_triple = ExponentialSmoothing(train_df, seasonal_periods=12, trend="add", seasonal="add")`
   - `model_triple_fit = model_triple.fit()`
   - Forecast: `forecast_triple = model_triple_fit.forecast(len(test_df))`
4. **Triple Exponential Smoothing Multiplicative (`ExponentialSmoothing`)**:
   - `model_triple_mul = ExponentialSmoothing(train_df, seasonal_periods=12, trend="add", seasonal="mul")`
   - `model_triple_fit_mul = model_triple_mul.fit()`
   - Forecast: `forecast_triple_mul = model_triple_fit_mul.forecast(len(test_df))`

### Phase 5: Accuracy Comparison (MAPE) (Cells 33 - 36)
- Evaluates out-of-sample accuracy using `mean_absolute_percentage_error`:
  - `mape_test_add = mean_absolute_percentage_error(test_df['CO2 Concentration'], forecast_triple)`
  - `mape_test_mul = mean_absolute_percentage_error(test_df['CO2 Concentration'], forecast_triple_mul)`
- **Aarchi's Finding**: "No difference in both methodologies" because the underlying seasonal oscillation height does not expand over time.

### Phase 6: Initial Stationarity Diagnostics (Cells 37 - 41)
- **ADF Test**:
  - `adfuller(df)`
  - Result: `p-value > 0.05` $\rightarrow$ Fail to reject $H_0$, series is non-stationary.
- **KPSS Test**:
  - `kpss(df)`
  - Result: `p-value = 0.01 < 0.05` $\rightarrow$ Reject $H_0$, series is non-stationary.
- **Conclusion**: Both tests confirm raw CO2 concentration is non-stationary and requires differencing.

---

## 4. Aarchi's Exact Python Code Reference

```python
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymannkendall as mk
import seaborn as sns
from sklearn.metrics import mean_absolute_percentage_error
from statsmodels.tsa.api import ExponentialSmoothing, Holt, SimpleExpSmoothing
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, kpss

# 1. Load and combine Year + Month
df = pd.read_csv('CO2 Concentration.xls')
df['year_month'] = (
    df['Year'].astype(str) + '-' + df['Month'].astype(str).str.zfill(2)
)
df['year_month'] = pd.to_datetime(df['year_month'])
df = df.set_index('year_month')
df.drop(columns=['Year', 'Month'], inplace=True)

# 2. Additive Seasonal Decomposition
result = seasonal_decompose(
    df[['CO2 Concentration']], model='additive', period=12
)
result.plot()
plt.show()

# 3. Mann-Kendall Test
mk_res = mk.original_test(df['CO2 Concentration'])
print(mk_res)

# 4. Train-Test Split (70/30)
train_df = df[: int(df.shape[0] * 0.7)]
test_df = df[int(df.shape[0] * 0.7) :]

# 5. Exponential Smoothing Models
model_single_fit = SimpleExpSmoothing(train_df).fit()
forecast_single = model_single_fit.forecast(len(test_df))

model_double_fit = Holt(train_df).fit()
forecast_double = model_double_fit.forecast(len(test_df))

model_triple_fit = ExponentialSmoothing(
    train_df, seasonal_periods=12, trend='add', seasonal='add'
).fit()
forecast_triple = model_triple_fit.forecast(len(test_df))

model_triple_fit_mul = ExponentialSmoothing(
    train_df, seasonal_periods=12, trend='add', seasonal='mul'
).fit()
forecast_triple_mul = model_triple_fit_mul.forecast(len(test_df))

# 6. Evaluation
mape_add = mean_absolute_percentage_error(
    test_df['CO2 Concentration'], forecast_triple
)
mape_mul = mean_absolute_percentage_error(
    test_df['CO2 Concentration'], forecast_triple_mul
)
print('MAPE Additive:', mape_add)
print('MAPE Multiplicative:', mape_mul)

# 7. Stationarity Tests
print('ADF Statistic & p-value:', adfuller(df['CO2 Concentration'])[:2])
print('KPSS Statistic & p-value:', kpss(df['CO2 Concentration'])[:2])
```
