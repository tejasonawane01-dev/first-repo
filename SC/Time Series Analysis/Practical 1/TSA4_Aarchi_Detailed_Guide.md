# Guide to Practical 4 (Yearly Revenue Analysis)

This document provides a breakdown of Practical 4 using Aarchi's time series framework on the `Yearly_revenue.xls` dataset.

---

## 📋 Table of Contents
1. [Overview & Objective](#1-overview--objective)
2. [Graph Breakdown & Visual Analysis](#2-graph-breakdown--visual-analysis)
3. [Step-by-Step Code Walkthrough](#3-step-by-step-code-walkthrough)

---

## 1. Overview & Objective
Annual data (non-seasonal series) requires integer indexing and trend-focused models like Holt's Linear model.

---

## 2. Graph Breakdown & Visual Analysis

### Graph 1: Line Plot (`sns.lineplot(data=df)`)
- Shows year-over-year revenue growth.

### Graph 2: Non-Seasonal Decomposition (`period=1`)
- Since data is annual, `period=1` eliminates seasonal sub-cycles.

### Graph 3: First Differencing Plot (`diff()`)
- Removes linear trend to test for stationarity.

---

## 3. Step-by-Step Code Walkthrough

```python
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from statsmodels.tsa.api import Holt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, kpss

# Load and integer index
df = pd.read_csv('Yearly_revenue.xls')
df['Year'] = range(1, len(df) + 1)
df = df.set_index('Year')

# Plot
sns.lineplot(data=df)

# Decompose
decomp = seasonal_decompose(df, model='multiplicative', period=1)
decomp.plot()

# Holt Linear Model
model_fit = Holt(df['Revenue']).fit()
print(model_fit.params)

# Stationarity
diff_rev = df['Revenue'].diff().dropna()
print('ADF p-value:', adfuller(diff_rev)[1])
print('KPSS p-value:', kpss(diff_rev)[1])
```
