# Comprehensive Guide to `TSA1-Aarchi.ipynb` (With Detailed Graph Explanations)

This document provides a complete, step-by-step breakdown of **`SC/Time Series Analysis/Practical 1/TSA1-Aarchi.ipynb`**, explaining every code line, statistical test, differencing stage, and **every single graph present in the notebook**.

---

## 📋 Table of Contents
1. [Notebook Overview](#1-notebook-overview)
2. [Detailed Explanation of ALL Graphs in `TSA1-Aarchi.ipynb`](#2-detailed-explanation-of-all-graphs-in-tsa1-aarchipynb)
   - [Graph 1: Raw AirPassengers Time Series Line Plot](#graph-1-raw-airpassengers-time-series-line-plot)
   - [Graph 2: Multiplicative Seasonal Decomposition (4 Sub-Plots)](#graph-2-multiplicative-seasonal-decomposition-4-sub-plots)
   - [Graph 3: Single Exponential Smoothing (SES) Forecast Plot](#graph-3-single-exponential-smoothing-ses-forecast-plot)
   - [Graph 4: Double Exponential Smoothing (Holt's Model) Forecast Plot](#graph-4-double-exponential-smoothing-holts-model-forecast-plot)
   - [Graph 5: Triple Exponential Smoothing (Holt-Winters Additive) Plot](#graph-5-triple-exponential-smoothing-holt-winters-additive-plot)
   - [Graph 6: Triple Exponential Smoothing (Holt-Winters Multiplicative) Plot](#graph-6-triple-exponential-smoothing-holt-winters-multiplicative-plot)
   - [Graphs 7 & 8: Raw Data ACF and PACF Plots](#graphs-7--8-raw-data-acf-and-pacf-plots)
   - [Graphs 9, 10 & 11: Non-Seasonal Differenced Series Line Plot, ACF & PACF](#graphs-9-10--11-non-seasonal-differenced-series-line-plot-acf--pacf)
   - [Graphs 12, 13 & 14: Seasonal Differenced Series Line Plot, ACF & PACF](#graphs-12-13--14-seasonal-differenced-series-line-plot-acf--pacf)
   - [Graphs 15, 16 & 17: Combined Differenced Series Line Plot, ACF & PACF](#graphs-15-16--17-combined-differenced-series-line-plot-acf--pacf)
3. [Statistical Tests Summary & Exam Rules](#3-statistical-tests-summary--exam-rules)
4. [Aarchi's Exact Code Reference](#4-aarchis-exact-code-reference)

---

## 1. Notebook Overview

`TSA1-Aarchi.ipynb` analyzes 144 monthly observations (1949–1960) of international airline passengers. The notebook explores time series decomposition, trend hypothesis testing, four exponential smoothing models, out-of-sample MAPE evaluation, and a 4-stage stationarity transformation pipeline.

---

## 2. Detailed Explanation of ALL Graphs in `TSA1-Aarchi.ipynb`

### Graph 1: Raw AirPassengers Time Series Line Plot
* **Code in Notebook**: `sns.lineplot(data); plt.ylabel("#Passengers")`
* **What it Shows**: A continuous line chart plotting monthly passenger count against time (1949 to 1960).
* **Visual Interpretation**:
  1. **Upward Trend**: The overall level increases steadily from ~100 to over 600 passengers.
  2. **Expanding Seasonal Oscillations**: The height of seasonal peaks (summer travel months) expands every year.
* **Aarchi's Notebook Observation**: *"We can see that the #passengers is increasing over time, with some seasonality, that is, with every year the trend is amplified."*

---

### Graph 2: Multiplicative Seasonal Decomposition (4 Sub-Plots)
* **Code in Notebook**: `result = seasonal_decompose(data[['#Passengers']], model='multiplicative', period=12); result.plot()`
* **What it Shows**: Breaks down the raw time series into 4 distinct components:
  1. **Observed (Panel 1)**: The original time series plot showing combined trend, seasonality, and noise.
  2. **Trend (Panel 2)**: Smooth, monotonically increasing curve representing long-term growth in passenger demand.
  3. **Seasonal (Panel 3)**: Repeating 12-month wave pattern. Identifies consistent peak demand during July/August and lowest demand during November/December.
  4. **Resid / Residuals (Panel 4)**: Random unexplained noise left after removing trend and seasonality.
* **Why Multiplicative?**: Multiplicative model ($Y_t = \text{Trend}_t \times \text{Seasonal}_t \times \text{Residual}_t$) is selected because seasonal oscillations grow proportionally as the trend rises.

---

### Graph 3: Single Exponential Smoothing (SES) Forecast Plot
* **Code in Notebook**: `plt.plot(data); plt.plot(model_single_fit.fittedvalues); plt.plot(forecast_single)`
* **What it Shows**: Original data vs. fitted in-sample values vs. out-of-sample forecast.
* **Visual Breakdown**:
  - **Fitted Values**: Follows actual data with a 1-step lag.
  - **Forecast Line**: A completely **flat horizontal line** into the test region.
* **Exam Key Concept**: SES only models level $\alpha$. Because it lacks trend ($\beta$) and seasonal ($\gamma$) components, its multi-step forecast is always a constant horizontal line.

---

### Graph 4: Double Exponential Smoothing (Holt's Model) Forecast Plot
* **Code in Notebook**: `plt.plot(data); plt.plot(model_double_fit.fittedvalues); plt.plot(forecast_double)`
* **What it Shows**: Original data vs. fitted values vs. Holt's linear forecast.
* **Visual Breakdown**:
  - **Forecast Line**: A **straight upward-sloping line** continuing into the test period.
* **Exam Key Concept**: Holt's Linear model fits level $\alpha$ and trend $\beta$. It captures the overall slope of passenger growth, but fails to capture seasonal peaks and troughs.

---

### Graph 5: Triple Exponential Smoothing (Holt-Winters Additive) Plot
* **Code in Notebook**: `plt.plot(data); plt.plot(model_triple_fit.fittedvalues); plt.plot(forecast_triple)`
* **What it Shows**: Holt-Winters Additive model forecast (`trend='add'`, `seasonal='add'`).
* **Visual Breakdown**:
  - **Forecast Line**: Displays repeating seasonal waves into the test period.
  - **Defect**: The amplitude (height) of forecasted peaks remains fixed at historical training heights, underestimating the higher peaks of 1958–1960.
* **Why it Happens**: Additive seasonality assumes seasonal fluctuations add a constant number of passengers regardless of how high the trend rises.

---

### Graph 6: Triple Exponential Smoothing (Holt-Winters Multiplicative) Plot
* **Code in Notebook**: `plt.plot(data); plt.plot(model_triple_fit_mul.fittedvalues); plt.plot(forecast_triple_mul)`
* **What it Shows**: Holt-Winters Multiplicative model forecast (`trend='add'`, `seasonal='mul'`).
* **Visual Breakdown**:
  - **Forecast Line**: Displays expanding seasonal waves that grow larger as the trend rises.
* **Conclusion**: Matches the actual test data almost perfectly, yielding the lowest **MAPE** score (~0.024 or 2.4% error).

---

### Graphs 7 & 8: Raw Data ACF and PACF Plots
* **Code in Notebook**: `plot_acf(data['#Passengers'])` and `plot_pacf(data['#Passengers'])`
* **Visual Breakdown**:
  - **ACF (Autocorrelation)**: Shows very slow, gradual linear decay across lags 1 to 40 with wave-like bumps at lag 12, 24, 36.
  - **PACF (Partial Autocorrelation)**: Extremely large dominant spike at Lag 1, while subsequent lags drop sharply.
* **Indication**: Slow decay in ACF confirms non-stationarity due to trend and strong seasonality.

---

### Graphs 9, 10 & 11: Non-Seasonal Differenced Series Line Plot, ACF & PACF
* **Code in Notebook**: `diff = data['#Passengers'].diff().dropna()`
* **Visual Breakdown**:
  - **Line Plot (Graph 9)**: Series centers around 0, showing trend removal.
  - **ACF Plot (Graph 10)**: Displays large prominent positive spikes at seasonal lags **12, 24, and 36**.
  - **PACF Plot (Graph 11)**: Shows negative spikes at seasonal intervals.
* **Indication**: Non-seasonal differencing removed trend, but strong **12-month seasonality remains**.

---

### Graphs 12, 13 & 14: Seasonal Differenced Series Line Plot, ACF & PACF
* **Code in Notebook**: `sdiff = data['#Passengers'].diff(periods=12).dropna()`
* **Visual Breakdown**:
  - **Line Plot (Graph 12)**: Seasonal oscillations removed.
  - **ACF Plot (Graph 13)**: Shows slow linear decay across initial lags (1, 2, 3, 4).
  - **PACF Plot (Graph 14)**: Dominant spike at Lag 1.
* **Indication**: Seasonal differencing removed seasonality, but **residual trend / drift remains**.

---

### Graphs 15, 16 & 17: Combined Differenced Series Line Plot, ACF & PACF
* **Code in Notebook**: `sddiff = sdiff.diff().dropna()`
* **Visual Breakdown**:
  - **Line Plot (Graph 15)**: Fluctuates randomly around mean 0 with stable variance.
  - **ACF Plot (Graph 16)**: Autocorrelation at Lag 0 is 1.0; all subsequent lags immediately fall inside the blue shaded 95% confidence interval.
  - **PACF Plot (Graph 17)**: All partial autocorrelations lie inside the confidence bounds.
* **Aarchi's Notebook Observation**: *"We finally have confirmed that the series is now stationary, via plots as well as tests."*

---

## 3. Statistical Tests Summary & Exam Rules

| Statistical Test | Code Expression | Null Hypothesis ($H_0$) | Decision Rule | Result on Raw Data | Result on Combined Diff (`sddiff`) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Mann-Kendall Test** | `mk.original_test(data)` | No monotonic trend | $p < 0.05 \rightarrow$ Trend exists | Trend Confirmed ($p < 0.05$) | N/A |
| **ADF Test** | `adfuller(series)` | Series is Non-Stationary | $p < 0.05 \rightarrow$ Stationary | Non-Stationary ($p = 0.99$) | **Stationary ($p < 0.05$)** |
| **KPSS Test** | `kpss(series)` | Series is Trend-Stationary | $p > 0.05 \rightarrow$ Stationary | Non-Stationary ($p = 0.01$) | **Stationary ($p > 0.05$)** |

---

## 4. Aarchi's Exact Code Reference

```python
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymannkendall as mk
import seaborn as sns
from sklearn.metrics import mean_absolute_percentage_error
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.api import ExponentialSmoothing, Holt, SimpleExpSmoothing
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, kpss

# 1. Load Data
data = pd.read_csv('AirPassengers.xls')
data['Month'] = pd.to_datetime(data['Month'])
data = data.set_index('Month')

# 2. Plot Raw Data (Graph 1)
sns.lineplot(data)
plt.ylabel('#Passengers')
plt.show()

# 3. Multiplicative Decomposition (Graph 2)
result = seasonal_decompose(
    data[['#Passengers']], model='multiplicative', period=12
)
result.plot()
plt.show()

# 4. Mann-Kendall Test
print(mk.original_test(data['#Passengers']))

# 5. Train Test Split
train_df = data[: int(data.shape[0] * 0.7)]
test_df = data[int(data.shape[0] * 0.7) :]

# 6. Fit Models (Graphs 3 - 6)
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

# 7. MAPE Evaluation
print(
    'MAPE Additive:',
    mean_absolute_percentage_error(test_df['#Passengers'], forecast_triple),
)
print(
    'MAPE Multiplicative:',
    mean_absolute_percentage_error(test_df['#Passengers'], forecast_triple_mul),
)

# 8. Differencing Pipeline (Graphs 7 - 17)
diff = data['#Passengers'].diff().dropna()
sdiff = data['#Passengers'].diff(12).dropna()
sddiff = sdiff.diff().dropna()

print('Final ADF p-value:', adfuller(sddiff)[1])
print('Final KPSS p-value:', kpss(sddiff)[1])

plot_acf(sddiff)
plt.show()
plot_pacf(sddiff)
plt.show()
```
