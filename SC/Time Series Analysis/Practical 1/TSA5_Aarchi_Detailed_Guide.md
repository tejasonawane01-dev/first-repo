# Guide to Practical 5 (Catfish Sales ARIMA Modeling)

This document provides a breakdown of Practical 5 using Aarchi's time series framework for Box-Jenkins **ARIMA** modeling on the `catfish.xls` dataset.

---

## 📋 Table of Contents
1. [Overview & Objective](#1-overview--objective)
2. [Graph Breakdown & Visual Analysis](#2-graph-breakdown--visual-analysis)
3. [Step-by-Step Code Walkthrough](#3-step-by-step-code-walkthrough)

---

## 1. Overview & Objective
Fits an **ARIMA(p, d, q)** model by identifying AR order $p$ and MA order $q$ from ACF and PACF plots after achieving stationarity via differencing ($d=1$).

---

## 2. Graph Breakdown & Visual Analysis

### Graph 1: Raw Catfish Sales Line Plot (`sns.lineplot(df)`)
- Shows monthly sales trend and volatility over time.

### Graph 2 & 3: Differenced Series ACF and PACF Plots
- **PACF Plot**: Used to identify AutoRegressive order $p$ (lag where PACF cuts off).
- **ACF Plot**: Used to identify Moving Average order $q$ (lag where ACF cuts off).

### Graph 4: Actual vs Forecasted Catfish Sales Plot
- Displays test period actual sales overlaid with `model_arima_fit.forecast()`.

---

## 3. Step-by-Step Code Walkthrough

```python
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import mean_absolute_percentage_error
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller, kpss

# Load and index
df = pd.read_csv('catfish.xls')
df['Date'] = pd.to_datetime(df['Date'])
df = df.set_index('Date')

# Stationarity & Differencing
diff_catfish = df['Total'].diff().dropna()
print('Differenced ADF p-value:', adfuller(diff_catfish)[1])

# Plot ACF & PACF
plot_acf(diff_catfish)
plt.show()
plot_pacf(diff_catfish)
plt.show()

# Train test split
train_df = df[: int(len(df) * 0.8)]
test_df = df[int(len(df) * 0.8) :]

# Fit ARIMA(1, 1, 1)
model_arima_fit = ARIMA(train_df['Total'], order=(1, 1, 1)).fit()
print(model_arima_fit.summary())

# Forecast
forecast = model_arima_fit.forecast(len(test_df))
print(
    'MAPE:',
    mean_absolute_percentage_error(test_df['Total'], forecast),
)

# Plot actual vs forecast
plt.plot(df['Total'], label='Original Data')
plt.plot(forecast, label='ARIMA Forecast')
plt.legend()
plt.show()
```
