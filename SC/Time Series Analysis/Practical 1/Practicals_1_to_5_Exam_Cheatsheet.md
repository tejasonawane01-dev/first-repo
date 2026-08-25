# Time Series Analysis — Master Exam Cheatsheet (Practicals 1 to 5)

This master document serves as your **exam study guide** for Time Series Analysis. It explains the core purpose of each of the 5 practicals, breaks down **why every line of code is written**, and provides the statistical rules required for exam questions and practical tests.

---

## 📌 Quick Summary of the 5 Practicals

| Practical | Dataset | Focus / Core Concept | Key Models / Tools Used |
| :--- | :--- | :--- | :--- |
| **Practical 1** | `AirPassengers.xls` | Exponential Smoothing & Full Stationarity Pipeline | `SimpleExpSmoothing`, `Holt`, `ExponentialSmoothing` (Add/Mul), `adfuller`, `kpss`, `diff()`, `diff(12)` |
| **Practical 2** | `CO2 Concentration.xls` | Data Preprocessing (Merging Year & Month) & Additive Modeling | `Year.astype(str) + '-' + Month`, `seasonal_decompose(model='additive')`, `pymannkendall` |
| **Practical 3** | `Electric_Production.xls` | Industrial Time Series & Multiplicative Forecasting Pipeline | `pd.to_datetime`, `seasonal_decompose(model='multiplicative')`, `ExponentialSmoothing` |
| **Practical 4** | `Yearly_revenue.xls` | Non-Seasonal / Annual Time Series | Integer indexing (`range(1, N+1)`), `period=1`, `Holt` linear model |
| **Practical 5** | `catfish.xls` | Box-Jenkins ARIMA & SARIMA Modeling | `adfuller`, `plot_acf` (MA order $q$), `plot_pacf` (AR order $p$), `ARIMA(train, order=(p, d, q))` |

---

## 💡 Why Each Line of Code is Written (Exam Reference)

### 1. Data Preprocessing & Date Indexing
```python
# Load raw CSV/XLS data
df = pd.read_csv('AirPassengers.xls')

# Convert string dates to pandas datetime objects
# WHY: Statsmodels models require recognized datetime objects to work with time frequencies.
df['Month'] = pd.to_datetime(df['Month'])

# Set the date column as the index
# WHY: Time series algorithms index data chronologically using the DataFrame index.
df = df.set_index('Month')
```

### 2. Handling Separate `Year` and `Month` Columns (Practical 2)
```python
# Combine numeric Year (e.g. 1958) and Month (e.g. 3) into '1958-03'
# WHY: 'zfill(2)' pads single digit months like '3' to '03' for standard ISO date formatting.
df['year_month'] = (
    df['Year'].astype(str) + '-' + df['Month'].astype(str).str.zfill(2)
)

df['year_month'] = pd.to_datetime(df['year_month'])
df = df.set_index('year_month')

# Drop redundant raw numeric columns to avoid confusion
df.drop(columns=['Year', 'Month'], inplace=True)
```

### 3. Time Series Decomposition
```python
from statsmodels.tsa.seasonal import seasonal_decompose

# Decompose time series into Trend, Seasonal, and Residual components
# WHY: model='multiplicative' is used when seasonal variations GROW as trend rises.
# WHY: model='additive' is used when seasonal variations remain CONSTANT height.
# WHY: period=12 is specified for monthly data (12 months per year).
decomp = seasonal_decompose(df['#Passengers'], model='multiplicative', period=12)
decomp.plot()
```

### 4. Mann-Kendall Trend Test
```python
import pymannkendall as mk

# Perform Mann-Kendall non-parametric test for monotonic trend
# WHY: Statistically confirms whether an overall upward/downward trend exists.
# H0 (Null Hypothesis): No monotonic trend in the series.
# H1 (Alternative): Monotonic trend exists.
# RULE: If p-value < 0.05, REJECT H0 -> Trend statistically exists.
result = mk.original_test(df['#Passengers'])
```

### 5. Sequential Train-Test Split
```python
# Calculate split point (e.g., 70% train, 30% test)
# WHY: Never shuffle time series data! Future values depend on past chronological order.
train_size = int(len(df) * 0.7)
train_df = df.iloc[:train_size]
test_df = df.iloc[train_size:]
```

---

### 6. Fitting Exponential Smoothing Models

```python
from statsmodels.tsa.api import ExponentialSmoothing, Holt, SimpleExpSmoothing

# Single Exponential Smoothing (SES)
# WHY: Used when data has NO trend and NO seasonality. Learns Level parameter alpha.
model_ses = SimpleExpSmoothing(train_df).fit()
forecast_ses = model_ses.forecast(len(test_df))

# Holt's Linear Double Exponential Smoothing
# WHY: Used when data HAS trend but NO seasonality. Learns Level alpha & Trend beta.
model_holt = Holt(train_df).fit()
forecast_holt = model_holt.forecast(len(test_df))

# Holt-Winters Triple Exponential Smoothing (Additive)
# WHY: Used when data has Level alpha, Trend beta, and CONSTANT seasonal amplitude gamma.
model_hw_add = ExponentialSmoothing(
    train_df, trend='add', seasonal='add', seasonal_periods=12
).fit()

# Holt-Winters Triple Exponential Smoothing (Multiplicative)
# WHY: Used when data has Level alpha, Trend beta, and EXPANDING seasonal amplitude gamma.
model_hw_mul = ExponentialSmoothing(
    train_df, trend='add', seasonal='mul', seasonal_periods=12
).fit()
```

---

### 7. Evaluation Metrics (MAPE & RMSE)
```python
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error

# Calculate Mean Absolute Percentage Error (MAPE)
# WHY: Measures percentage error of forecast vs actual test data. Lower is better.
# CRITICAL EXAM RULE: Always pass actual test values FIRST, forecast SECOND!
mape = mean_absolute_percentage_error(test_df['#Passengers'], forecast_hw_mul)

# Root Mean Squared Error (RMSE)
rmse = mean_squared_error(
    test_df['#Passengers'], forecast_hw_mul, squared=False
)
```

---

### 8. Statistical Stationarity Testing (ADF & KPSS Rules)

```python
from statsmodels.tsa.stattools import adfuller, kpss

# 1. Augmented Dickey-Fuller (ADF) Test
# H0: Series is Non-Stationary (Has a unit root).
# H1: Series is Stationary.
# EXAM RULE: ADF p-value < 0.05 -> Reject H0 -> STATIONARY.
adf_p = adfuller(df['#Passengers'])[1]

# 2. KPSS Test
# H0: Series is Stationary / Trend Stationary.
# H1: Series is Non-Stationary.
# EXAM RULE: KPSS p-value > 0.05 -> Fail to reject H0 -> STATIONARY.
kpss_p = kpss(df['#Passengers'], regression='c')[1]
```

> [!IMPORTANT]
> **Stationarity Golden Rule for Exams**:
> Data is **fully stationary** ONLY when:
> 1. **ADF p-value < 0.05** (Passes ADF)
> 2. **KPSS p-value > 0.05** (Passes KPSS)

---

### 9. Multi-Stage Differencing Pipeline
```python
# Non-Seasonal Differencing (d=1) -> Removes Trend
diff1 = df['#Passengers'].diff().dropna()

# Seasonal Differencing (D=1, period=12) -> Removes Seasonality
sdiff = df['#Passengers'].diff(12).dropna()

# Combined Seasonal & Non-Seasonal Differencing -> Achieves Full Stationarity
sddiff = sdiff.diff().dropna()
```

---

### 10. Box-Jenkins ARIMA Modeling (Practical 5)
```python
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA

# Plot ACF and PACF to determine orders p and q
# PACF cutoff lag -> AutoRegressive order p
# ACF cutoff lag  -> Moving Average order q
fig, axes = plt.subplots(1, 2, figsize=(14, 4))
plot_acf(diff_data, ax=axes[0], title='ACF for MA(q)')
plot_pacf(diff_data, ax=axes[1], title='PACF for AR(p)')

# Fit ARIMA(p, d, q) Model
# p = AR order, d = differencing count, q = MA order
model_arima = ARIMA(train_df, order=(1, 1, 1))
model_arima_fit = model_arima.fit()

# View summary table (Coefficients, p-values, AIC score)
print(model_arima_fit.summary())

# Forecast future values
forecast = model_arima_fit.forecast(steps=len(test_df))
```

---

## 📁 Accessing Your 5 Clean Jupyter Notebook Files

All 5 exam-guide `.ipynb` notebooks have been generated without editing your original practical files. You can open and run them directly:

1. 📄 [`Practical_1_Exam_Guide.ipynb`](file:///c:/Users/tejas/Sem_3/SC/Time%20Series%20Analysis/Practical%201/Practical_1_Exam_Guide.ipynb) (AirPassengers - SES, Holt, Holt-Winters, Stationarity Pipeline)
2. 📄 [`Practical_2_Exam_Guide.ipynb`](file:///c:/Users/tejas/Sem_3/SC/Time%20Series%20Analysis/Practical%201/Practical_2_Exam_Guide.ipynb) (CO2 Concentration - Year/Month Merging, Additive Model, Stationarity)
3. 📄 [`Practical_3_Exam_Guide.ipynb`](file:///c:/Users/tejas/Sem_3/SC/Time%20Series%20Analysis/Practical%201/Practical_3_Exam_Guide.ipynb) (Electric Production - Multiplicative Decomposition & Holt-Winters)
4. 📄 [`Practical_4_Exam_Guide.ipynb`](file:///c:/Users/tejas/Sem_3/SC/Time%20Series%20Analysis/Practical%201/Practical_4_Exam_Guide.ipynb) (Yearly Revenue - Non-Seasonal Integer Indexing & Holt Linear Model)
5. 📄 [`Practical_5_Exam_Guide.ipynb`](file:///c:/Users/tejas/Sem_3/SC/Time%20Series%20Analysis/Practical%201/Practical_5_Exam_Guide.ipynb) (Catfish Sales - ACF/PACF Order Selection & ARIMA Modeling)
