# Complete Guide to `TSA1-Aarchi.ipynb` (Time Series Analysis - Practical 1)

This document provides a complete, step-by-step breakdown of everything performed in **`SC/Time Series Analysis/Practical 1/TSA1-Aarchi.ipynb`**. It covers theoretical concepts, statistical tests, model comparisons, differencing steps for stationarity, and ready-to-use Python code reference.

---

## 📋 Table of Contents
1. [Overview & Objectives](#1-overview--objectives)
2. [Notebook Architecture & Workflow Overview](#2-notebook-architecture--workflow-overview)
3. [Key Concepts & Statistical Tests](#3-key-concepts--statistical-tests)
4. [Section-by-Section Breakdown](#4-section-by-section-breakdown)
   - [Phase 1: Data Preprocessing & Initial Visualization](#phase-1-data-preprocessing--initial-visualization)
   - [Phase 2: Decomposition & Mann-Kendall Trend Test](#phase-2-decomposition--mann-kendall-trend-test)
   - [Phase 3: Train-Test Split](#phase-3-train-test-split)
   - [Phase 4: Exponential Smoothing Forecasting Models](#phase-4-exponential-smoothing-forecasting-models)
   - [Phase 5: Accuracy Evaluation (MAPE)](#phase-5-accuracy-evaluation-mape)
   - [Phase 6: Stationarity Testing & Multi-Stage Differencing](#phase-6-stationarity-testing--multi-stage-differencing)
5. [Summary Matrix of Stationarity Transformation](#5-summary-matrix-of-stationarity-transformation)
6. [Complete Code Cheatsheet](#6-complete-code-cheatsheet)

---

## 1. Overview & Objectives

In `TSA1-Aarchi.ipynb`, time series analysis and forecasting are performed on the classic **AirPassengers** dataset (monthly international airline passengers from 1949 to 1960, containing 144 observations).

### Main Goals:
1. **Analyze Trend & Seasonality**: Confirm monotonic trend using statistical tests (Mann-Kendall test) and seasonal decomposition.
2. **Build & Compare Forecasting Models**: Fit Exponential Smoothing models (Single, Double/Holt's, and Triple/Holt-Winters Additive & Multiplicative) and evaluate their performance using MAPE.
3. **Achieve Stationarity for Box-Jenkins / ARIMA Preparation**: Test stationarity with **ADF** and **KPSS** tests, plot **ACF** and **PACF**, and apply non-seasonal and seasonal differencing until full stationarity is established.

---

## 2. Notebook Architecture & Workflow Overview

```
                      AirPassengers Dataset (144 rows)
                                     │
                     ┌───────────────┴───────────────┐
                     ▼                               ▼
       Part A: Exponential Smoothing    Part B: Stationarity Analysis
                     │                               │
            Decomposition & Trend               Stationarity Tests
            (Mann-Kendall Test)             (ADF, KPSS, ACF, PACF)
                     │                               │
             Train/Test Split               Multi-Stage Differencing
              (70% / 30%)               ┌────────────┼────────────┐
                     │                  ▼            ▼            ▼
             Model Building         Non-Seasonal  Seasonal    Combined
          ┌──────────┼──────────┐    diff()     diff(12)   sdiff.diff()
          ▼          ▼          ▼       │            │            │
         SES       Holt's  Holt-Winters  ADF/KPSS    ADF/KPSS   ADF/KPSS
                            (Add & Mul) ACF/PACF    ACF/PACF   ACF/PACF
                                │                                 │
                            Evaluation                    Fully Stationary
                           (MAPE Test)                    (Ready for ARIMA)
```

---

## 3. Key Concepts & Statistical Tests

### A. Exponential Smoothing Family
* **Single Exponential Smoothing (SES)**: Fits level $\alpha$. Used when there is no trend or seasonality. Produces flat forecasts.
* **Double Exponential Smoothing (Holt's Linear)**: Fits level $\alpha$ and trend $\beta$. Captures linear trends, but ignores seasonality.
* **Triple Exponential Smoothing (Holt-Winters)**: Fits level $\alpha$, trend $\beta$, and seasonality $\gamma$.
  * **Additive (`trend='add'`, `seasonal='add'`)**: Used when seasonal variations are constant over time.
  * **Multiplicative (`trend='add'`, `seasonal='mul'`)**: Used when seasonal variations increase proportionally with the trend.

### B. Mann-Kendall Trend Test
* Non-parametric test used to check if a monotonic upward or downward trend exists in time series data.
* **$H_0$ (Null Hypothesis)**: No monotonic trend in the series.
* **$H_1$ (Alternative Hypothesis)**: A monotonic trend exists.

### C. Stationarity Tests: Dual Testing Approach (ADF & KPSS)

| Test | Null Hypothesis ($H_0$) | Alternative Hypothesis ($H_1$) | Stationary Condition |
| :--- | :--- | :--- | :--- |
| **ADF Test** (Augmented Dickey-Fuller) | Series is **Non-Stationary** (Has a unit root) | Series is **Stationary** | **p-value < 0.05** (Reject $H_0$) |
| **KPSS Test** (Kwiatkowski-Phillips-Schmidt-Shin) | Series is **Trend Stationary** (No unit root) | Series is **Non-Stationary** | **p-value > 0.05** (Fail to reject $H_0$) |

> [!IMPORTANT]
> A series is strictly confirmed as stationary only when **both** conditions are met:
> 1. **ADF p-value < 0.05** (Rejects unit root)
> 2. **KPSS p-value > 0.05** (Fails to reject stationarity)

### D. Autocorrelation (ACF) & Partial Autocorrelation (PACF)
* **ACF (Autocorrelation Function)**: Measures total correlation between series and its lagged values (including indirect effects).
* **PACF (Partial Autocorrelation Function)**: Measures direct correlation between series and its lagged values, removing intermediate lag effects.

---

## 4. Section-by-Section Breakdown

### Phase 1: Data Preprocessing & Initial Visualization
* Reads `AirPassengers.xls`.
* Converts `Month` column to datetime object: `pd.to_datetime(data['Month'])`.
* Sets `Month` as index: `data.set_index('Month')`.
* Shape: `(144, 1)`.
* Visualizes `#Passengers` over time using `sns.lineplot()`.
* **Observation**: Noticeable upward trend with growing seasonal oscillations.

### Phase 2: Decomposition & Mann-Kendall Trend Test
* Multiplicative seasonal decomposition performed using `seasonal_decompose(data[['#Passengers']], model='multiplicative', period=12)`.
* Monotonic trend verified using `pymannkendall`:
  * Code: `mk.original_test(data['#Passengers'])`
  * **Result**: Confirms significant upward trend ($p < 0.05$).

### Phase 3: Train-Test Split
* Sequential 70% / 30% split (time-series data must NOT be shuffled):
  * `train_df`: First 70% (~100 months)
  * `test_df`: Remaining 30% (~44 months)

### Phase 4: Exponential Smoothing Forecasting Models
Fits 4 models on `train_df` and forecasts for the length of `test_df`:
1. **Single Exponential Smoothing (`SimpleExpSmoothing`)**:
   * Learns level parameter $\alpha$.
   * Forecast is flat horizontal line.
2. **Double Exponential Smoothing (`Holt`)**:
   * Learns level $\alpha$ and trend $\beta$.
   * Forecast captures upward slope, but misses seasonal waves.
3. **Holt-Winters Additive (`ExponentialSmoothing(..., trend='add', seasonal='add', seasonal_periods=12)`)**:
   * Captures upward trend and fixed-amplitude seasonal cycles.
4. **Holt-Winters Multiplicative (`ExponentialSmoothing(..., trend='add', seasonal='mul', seasonal_periods=12)`)**:
   * Captures upward trend and expanding seasonal oscillations.

### Phase 5: Accuracy Evaluation (MAPE)
Evaluates forecast accuracy against actual `test_df['#Passengers']` using `mean_absolute_percentage_error`:
* `mape_test_add`: Additive Holt-Winters MAPE.
* `mape_test_mul`: Multiplicative Holt-Winters MAPE.
* **Conclusion**: Multiplicative model yields lower MAPE because seasonal amplitude expands as the trend rises.

---

### Phase 6: Stationarity Testing & Multi-Stage Differencing

#### Stage 1: Raw Un-differenced Data
* **ADF Test**: `p-value > 0.05` $\rightarrow$ Fails to reject $H_0$ (Non-stationary).
* **KPSS Test**: `p-value < 0.05` $\rightarrow$ Rejects $H_0$ (Non-stationary).
* **ACF / PACF Plots**: ACF shows slow linear decay typical of non-stationary series with trend and seasonality.

#### Stage 2: First Non-Seasonal Differencing `diff()`
* Code: `diff = data['#Passengers'].diff().dropna()`
* **ADF Test**: `p-value < 0.05` $\rightarrow$ Stationary according to ADF.
* **KPSS Test**: `p-value < 0.05` $\rightarrow$ Non-stationary according to KPSS.
* **ACF / PACF Plots**: Significant spikes remaining at seasonal lags (12, 24).
* **Conclusion**: Non-seasonal differencing removed the trend, but strong seasonality remains.

#### Stage 3: First Seasonal Differencing `diff(12)`
* Code: `sdiff = data['#Passengers'].diff(periods=12).dropna()`
* **ADF Test**: `p-value < 0.05` $\rightarrow$ Stationary according to ADF.
* **KPSS Test**: `p-value < 0.05` $\rightarrow$ Non-stationary according to KPSS.
* **ACF / PACF Plots**: Significant autocorrelations still present.
* **Conclusion**: Seasonal differencing removed seasonality, but residual trend/drift remains.

#### Stage 4: Combined Seasonal & Non-Seasonal Differencing `sdiff.diff()`
* Code: `sddiff = sdiff.diff().dropna()` (Non-seasonal diff applied on seasonal diff series).
* **ADF Test**: `p-value < 0.05` $\rightarrow$ **PASS** (Stationary).
* **KPSS Test**: `p-value > 0.05` $\rightarrow$ **PASS** (Trend Stationary / Stationary).
* **ACF / PACF Plots**: Autocorrelations quickly drop to near zero within confidence bounds.
* **Conclusion**: The series is now **fully stationary** and ready for ARIMA modeling ($d=1, D=1, s=12$).

---

## 5. Summary Matrix of Stationarity Transformation

| Transformation Stage | Code Expression | ADF Test ($p < 0.05$?) | KPSS Test ($p > 0.05$?) | Stationary Status |
| :--- | :--- | :--- | :--- | :--- |
| **Raw Data** | `data['#Passengers']` | ❌ No ($p > 0.05$) | ❌ No ($p < 0.05$) | **Non-Stationary** |
| **Non-Seasonal Diff** | `.diff()` | ✅ Yes | ❌ No ($p < 0.05$) | **Partially Stationary (Seasonality remains)** |
| **Seasonal Diff** | `.diff(12)` | ✅ Yes | ❌ No ($p < 0.05$) | **Partially Stationary (Trend remains)** |
| **Combined Diff** | `.diff(12).diff()` | ✅ Yes | ✅ Yes ($p > 0.05$) | **Fully Stationary** |

---

## 6. Complete Code Cheatsheet

Below are all the key Python code blocks used in `TSA1-Aarchi.ipynb` for quick copy-pasting and study:

### 1. Data Imports & Loading
```python
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose

# Read and index data
data = pd.read_csv('AirPassengers.xls')
data['Month'] = pd.to_datetime(data['Month'])
data = data.set_index('Month')
```

### 2. Mann-Kendall Trend Test
```python
import pymannkendall as mk

# Test for monotonic trend
mk_result = mk.original_test(data['#Passengers'])
print(mk_result)
```

### 3. Train-Test Split (70/30)
```python
train_size = int(len(data) * 0.7)
train_df = data.iloc[:train_size]
test_df = data.iloc[train_size:]
```

### 4. Exponential Smoothing Models
```python
from sklearn.metrics import mean_absolute_percentage_error
from statsmodels.tsa.api import ExponentialSmoothing, Holt, SimpleExpSmoothing

# 1. Single Exponential Smoothing
model_ses = SimpleExpSmoothing(train_df).fit()
forecast_ses = model_ses.forecast(len(test_df))

# 2. Holt's Linear (Double Exponential)
model_holt = Holt(train_df).fit()
forecast_holt = model_holt.forecast(len(test_df))

# 3. Holt-Winters Additive
model_hw_add = ExponentialSmoothing(
    train_df, trend='add', seasonal='add', seasonal_periods=12
).fit()
forecast_hw_add = model_hw_add.forecast(len(test_df))

# 4. Holt-Winters Multiplicative
model_hw_mul = ExponentialSmoothing(
    train_df, trend='add', seasonal='mul', seasonal_periods=12
).fit()
forecast_hw_mul = model_hw_mul.forecast(len(test_df))

# Evaluate MAPE
mape_add = mean_absolute_percentage_error(
    test_df['#Passengers'], forecast_hw_add
)
mape_mul = mean_absolute_percentage_error(
    test_df['#Passengers'], forecast_hw_mul
)
print('Additive MAPE:', mape_add)
print('Multiplicative MAPE:', mape_mul)
```

### 5. Statistical Stationarity Testing & Differencing Pipeline
```python
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller, kpss


def test_stationarity(series, name='Series'):
  print(f'=== {name} ===')

  # ADF Test
  adf_res = adfuller(series.dropna())
  print(f'ADF Statistic: {adf_res[0]:.4f}, p-value: {adf_res[1]:.4f}')

  # KPSS Test
  kpss_res = kpss(series.dropna(), regression='c')
  print(f'KPSS Statistic: {kpss_res[0]:.4f}, p-value: {kpss_res[1]:.4f}\n')


# 1. Raw Series
test_stationarity(data['#Passengers'], 'Raw Data')

# 2. Non-Seasonal Differencing
diff = data['#Passengers'].diff().dropna()
test_stationarity(diff, 'Non-Seasonal Diff d=1')

# 3. Seasonal Differencing
sdiff = data['#Passengers'].diff(12).dropna()
test_stationarity(sdiff, 'Seasonal Diff D=1')

# 4. Combined Differencing (Seasonal + Non-Seasonal)
sddiff = sdiff.diff().dropna()
test_stationarity(sddiff, 'Combined Seasonal & Non-Seasonal Diff')

# Plot ACF & PACF of fully stationary series
fig, axes = plt.subplots(1, 2, figsize=(16, 4))
plot_acf(sddiff, ax=axes[0], lags=40)
plot_pacf(sddiff, ax=axes[1], lags=40)
plt.show()
```

---

## 7. Summary & Final Conclusions

1. **AirPassengers Dataset Characteristics**: Displays strong upward trend and expanding seasonal oscillations (period = 12 months).
2. **Best Forecasting Model**: **Holt-Winters Multiplicative** model out-performs Additive, SES, and Holt's models because it dynamically scales seasonal variation with the trend level.
3. **Stationarity Requirement**: Neither single non-seasonal differencing (`diff()`) nor single seasonal differencing (`diff(12)`) alone achieves full stationarity under both ADF and KPSS tests.
4. **Final Differencing**: Applying both seasonal and first differencing (`sddiff = data['#Passengers'].diff(12).diff().dropna()`) yields a fully stationary series, satisfying ADF ($p < 0.05$) and KPSS ($p > 0.05$) tests, ready for SARIMA modeling.
