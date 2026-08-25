# Guide to Practical 3 (Electricity Production Analysis)

This document provides a breakdown of Practical 3 using Aarchi's time series framework on the `Electric_Production.xls` dataset.

---

## 📋 Table of Contents
1. [Overview & Objective](#1-overview--objective)
2. [Graph Breakdown & Visual Analysis](#2-graph-breakdown--visual-analysis)
3. [Step-by-Step Code Walkthrough](#3-step-by-step-code-walkthrough)

---

## 1. Overview & Objective
Industrial electricity production data contains strong annual seasonality and long-term economic trends.

---

## 2. Graph Breakdown & Visual Analysis

### Graph 1: Line Plot (`sns.lineplot(df)`)
- Shows long-term growth in electricity demand along with winter and summer peaks.

### Graph 2: Decomposition Plot (`seasonal_decompose(..., model='multiplicative')`)
- **Observed**: Raw industrial production index.
- **Trend**: Smooth trend showing economic cycles.
- **Seasonal**: Annual 12-month cycle representing winter heating and summer cooling demand peaks.
- **Resid**: Unexplained noise.

### Graph 3: Holt-Winters Multiplicative Forecast Plot
- Shows forecast waves that scale proportionally with future production growth.

### Graph 4: ACF & PACF Plots of Differenced Series (`sdiff.diff()`)
- Confirms stationarity ($p_{\text{ADF}} < 0.05$ and $p_{\text{KPSS}} > 0.05$).

---

## 3. Step-by-Step Code Walkthrough

```python
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymannkendall as mk
import seaborn as sns
from sklearn.metrics import mean_absolute_percentage_error
from statsmodels.tsa.api import ExponentialSmoothing
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, kpss

# Load and index
df = pd.read_csv('Electric_Production.xls')
df['DATE'] = pd.to_datetime(df['DATE'])
df = df.set_index('DATE')

# Decompose
decomp = seasonal_decompose(
    df[['IPG2211A2N']], model='multiplicative', period=12
)
decomp.plot()

# Train test split
train_df = df[: int(len(df) * 0.7)]
test_df = df[int(len(df) * 0.7) :]

# Holt-Winters Multiplicative
model_fit = ExponentialSmoothing(
    train_df['IPG2211A2N'], trend='add', seasonal='mul', seasonal_periods=12
).fit()
forecast = model_fit.forecast(len(test_df))

# MAPE
print(
    'MAPE:',
    mean_absolute_percentage_error(test_df['IPG2211A2N'], forecast),
)

# Stationarity
sddiff = df['IPG2211A2N'].diff(12).diff().dropna()
print('Differenced ADF p-value:', adfuller(sddiff)[1])
print('Differenced KPSS p-value:', kpss(sddiff)[1])
```
