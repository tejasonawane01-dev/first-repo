# Time Series Analysis: Practical 1 — Comprehensive Guide & Code Reference

This document provides a complete step-by-step breakdown of everything performed in **`practical 1.ipynb`**, along with a **Code Reference & Cheatsheet** containing essential functions, syntax, and easily forgotten details.

---

## 1. Overview & Objective of Practical 1

In this practical, you are performing end-to-end **Time Series Analysis and Forecasting** on the classic **AirPassengers** dataset (monthly totals of international airline passengers from 1949 to 1960).

### Key Concepts Covered:
1. **Data Preprocessing**: Converting date columns to datetime formats and setting datetime indices for time-indexed DataFrames.
2. **Train-Test Split**: Splitting chronological time series data into training and testing sets (without shuffling).
3. **Classical Time Series Decomposition**: Decomposing time series into **Trend**, **Seasonal**, and **Residual (Observed/Noise)** components.
4. **Exponential Smoothing Models**:
   - **Single Exponential Smoothing (SES)**: Suitable for time series without trend or seasonality.
   - **Double Exponential Smoothing (Holt's Linear)**: Captures level and trend.
   - **Triple Exponential Smoothing (Holt-Winters)**: Captures level, trend, and seasonality (evaluated for both **Additive** and **Multiplicative** variations).
5. **Model Evaluation & Forecasting**: Generating multi-step forecasts on the test dataset and evaluating accuracy using **MAPE (Mean Absolute Percentage Error)**.

---

## 2. Step-by-Step Breakdown of What You Are Doing

### Step 1: Library Imports
You import the necessary data processing, plotting, and statsmodels libraries:
- `pandas` & `numpy` for data manipulation.
- `matplotlib.pyplot` & `seaborn` for visualization.
- `statsmodels.tsa.seasonal.seasonal_decompose` for time series decomposition.
- `statsmodels.tsa.api` (`SimpleExpSmoothing`, `Holt`, `ExponentialSmoothing`) for exponential smoothing models.
- `sklearn.metrics` (`mean_absolute_percentage_error`, `mean_squared_error`, `mean_absolute_error`) for model evaluation.

### Step 2: Loading & Preprocessing Time Series Data
- You read `AirPassengers.xls` (or `.csv`) using `pd.read_csv()`.
- You convert the `Month` column into a Pandas `datetime64` object via `pd.to_datetime()`.
- You set `Month` as the DataFrame index via `.set_index("Month")`. This is crucial because `statsmodels` requires a proper DatetimeIndex to infer sampling frequency (`MS` = Month Start).

### Step 3: Exploratory Visualization & Train-Test Split
- You plot the raw time series `#Passengers` against time to observe the upward trend and expanding seasonal oscillations.
- You split the dataset sequentially into:
  - **Training Set (`train_df`)**: First ~100 months (1949 to April 1957).
  - **Test Set (`test_df`)**: Remaining months (May 1957 to December 1960).

### Step 4: Time Series Decomposition
- You decompose the series into **Observed**, **Trend**, **Seasonal**, and **Residual** components using `seasonal_decompose()`.
- You plot the individual components to verify seasonal periodicity (period = 12 months).

### Step 5: Single Exponential Smoothing (SES)
- **Model**: `SimpleExpSmoothing(train_df)`
- Fits level parameter $\alpha$ (alpha).
- Produces a flat horizontal forecast because it does not model trend or seasonality.

### Step 6: Double Exponential Smoothing (Holt's Linear Model)
- **Model**: `Holt(train_df)`
- Fits level ($\alpha$) and trend ($\beta$) parameters.
- Produces a linear (sloped) forecast that captures the upward trend, but fails to capture seasonal peaks and troughs.

### Step 7: Triple Exponential Smoothing — Additive (Holt-Winters Additive)
- **Model**: `ExponentialSmoothing(train_df, seasonal='add', seasonal_periods=12, trend='add')`
- Fits level ($\alpha$), trend ($\beta$), and additive seasonal components ($\gamma$).
- Assumes seasonal fluctuations remain constant in size as the trend rises.

### Step 8: Triple Exponential Smoothing — Multiplicative (Holt-Winters Multiplicative)
- **Model**: `ExponentialSmoothing(train_df, seasonal='mul', seasonal_periods=12, trend='add')`
- Fits level ($\alpha$), trend ($\beta$), and multiplicative seasonal components ($\gamma$).
- Best fits data where seasonal oscillations grow proportionally with the overall trend level (like AirPassengers).

### Step 9: Forecast Evaluation
- You generate out-of-sample forecasts for `len(test_df)` steps.
- You compute performance metrics using `mean_absolute_percentage_error(test_df['#Passengers'], forecast)`.

---

## 3. Code Reference & Cheatsheet (Easy-to-Forget Snippets)

> [!IMPORTANT]
> Keep this code reference handy for practical exams and future time series tasks.

### 1. Data Preprocessing & Date Indexing
```python
import pandas as pd

# Load CSV data
df = pd.read_csv('AirPassengers.xls')

# Convert text column to datetime
df['Month'] = pd.to_datetime(df['Month'])

# Set datetime column as index (Essential for time series models!)
df = df.set_index('Month')
```

### 2. Time Series Train-Test Split
```python
# Sequential split (Do NOT shuffle time series data!)
train_df = df.iloc[:100]  # First 100 observations
test_df = df.iloc[100:]  # Remaining observations
```

### 3. Time Series Decomposition
```python
from statsmodels.tsa.seasonal import seasonal_decompose

# Decompose into Trend, Seasonal (period=12 for monthly), and Residuals
decomposition = seasonal_decompose(df['#Passengers'], model='additive', period=12)

# Plot components
fig = decomposition.plot()
```

---

### 4. Exponential Smoothing Models (Statsmodels)

#### A. Single Exponential Smoothing (SES)
```python
from statsmodels.tsa.api import SimpleExpSmoothing

# Fit model
model_single = SimpleExpSmoothing(train_df)
model_single_fit = model_single.fit()

# Forecast for test period length
forecast_single = model_single_fit.forecast(len(test_df))

# Inspect learned parameters (alpha, etc.)
print(model_single_fit.params)
```

#### B. Double Exponential Smoothing (Holt's Linear)
```python
from statsmodels.tsa.api import Holt

# Fit model with level and trend
model_double = Holt(train_df)
model_double_fit = model_double.fit()

# Forecast
forecast_double = model_double_fit.forecast(len(test_df))
```

#### C. Triple Exponential Smoothing (Holt-Winters Additive)
```python
from statsmodels.tsa.api import ExponentialSmoothing

# Fit model with level, trend, and additive seasonality
model_triple_add = ExponentialSmoothing(
    train_df, trend='add', seasonal='add', seasonal_periods=12
)
model_triple_fit_add = model_triple_add.fit()

# Forecast
forecast_triple_add = model_triple_fit_add.forecast(len(test_df))
```

#### D. Triple Exponential Smoothing (Holt-Winters Multiplicative)
```python
from statsmodels.tsa.api import ExponentialSmoothing

# Fit model with level, trend, and multiplicative seasonality
model_triple_mul = ExponentialSmoothing(
    train_df, trend='add', seasonal='mul', seasonal_periods=12
)
model_triple_fit_mul = model_triple_mul.fit()

# Forecast
forecast_triple_mul = model_triple_fit_mul.forecast(len(test_df))
```

---

### 5. Evaluation Metrics (`sklearn.metrics`)

> [!WARNING]
> Always pass `y_true` (actual test data) first and `y_pred` (forecasted values) second into `sklearn` metric functions: `func(y_true, y_pred)`.

```python
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
)

# Mean Absolute Percentage Error (MAPE)
mape = mean_absolute_percentage_error(test_df['#Passengers'], forecast_triple_mul)
print('MAPE:', mape)

# Mean Squared Error (MSE) and Root Mean Squared Error (RMSE)
mse = mean_squared_error(test_df['#Passengers'], forecast_triple_mul)
rmse = mean_squared_error(
    test_df['#Passengers'], forecast_triple_mul, squared=False
)

# Mean Absolute Error (MAE)
mae = mean_absolute_error(test_df['#Passengers'], forecast_triple_mul)
```

---

### 6. Plotting Fitted Values and Forecasts
```python
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 5))
plt.plot(df['#Passengers'], label='Original Data')
plt.plot(model_triple_fit_mul.fittedvalues, label='In-Sample Fitted Values')
plt.plot(forecast_triple_mul, label='Out-of-Sample Forecast', color='red')
plt.xlabel('Year')
plt.ylabel('Passengers')
plt.title('Holt-Winters Multiplicative Forecast vs Actuals')
plt.legend()
plt.show()
```

---

## 4. Summary Table of Smoothing Models

| Model | Trend | Seasonality | Best Used For | Code Call |
| :--- | :--- | :--- | :--- | :--- |
| **Simple Exp Smoothing (SES)** | None | None | Data with level only | `SimpleExpSmoothing(train_df)` |
| **Holt's Linear Smoothing** | Additive / Damped | None | Data with trend, no seasonality | `Holt(train_df)` |
| **Holt-Winters Additive** | Additive | Additive | Constant seasonal variations | `ExponentialSmoothing(..., seasonal='add')` |
| **Holt-Winters Multiplicative** | Additive | Multiplicative | Seasonal variations growing with trend | `ExponentialSmoothing(..., seasonal='mul')` |
