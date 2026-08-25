const fs = require('fs');
const path = require('path');

function mc(type, lines) {
  return {
    cell_type: type,
    metadata: {},
    source: lines.map((l, i) => i === lines.length - 1 ? l : l + '\n'),
    outputs: []
  };
}

function nb(cells) {
  return { cells, metadata: { language_info: { name: 'python' } }, nbformat: 4, nbformat_minor: 2 };
}

const out = path.join(__dirname, 'SC/Time Series Analysis/guide practical 1 using GPT');

// ============================================================================
// PRACTICAL 1: AirPassengers
// ============================================================================
const p1 = [
  mc('markdown', [
    '# Practical 1: Time Series Analysis & Forecasting',
    '## Dataset: AirPassengers.xls',
    '## Objective: Perform decomposition, exponential smoothing forecasting, and stationarity analysis'
  ]),

  // --- IMPORTS ---
  mc('code', [
    'import numpy as np',
    'import pandas as pd',
    'import seaborn as sns',
    'import matplotlib.pyplot as plt',
    'import os',
    'from statsmodels.tsa.seasonal import seasonal_decompose'
  ]),

  // --- LOAD DATA ---
  mc('code', [
    'data = pd.read_csv("AirPassengers.xls")'
  ]),
  mc('code', [
    'data.head(5)'
  ]),
  mc('code', [
    'data.shape'
  ]),

  // --- DATETIME INDEXING ---
  mc('code', [
    '#convert column year in date time format (truncating to month)',
    'data[\'Month\'] = pd.to_datetime(data[\'Month\'])',
    'data = data.set_index(\'Month\')'
  ]),
  mc('code', [
    'data.head(5)'
  ]),
  mc('code', [
    'data.shape'
  ]),

  // --- LINE PLOT ---
  mc('code', [
    'sns.lineplot(data)',
    'plt.ylabel("#Passengers")'
  ]),
  mc('markdown', [
    '### 📊 How to Read This Graph (Line Plot):',
    '- **X-axis** = Time (1949 to 1960)',
    '- **Y-axis** = Number of airline passengers per month',
    '- **What to observe:**',
    '  1. The line goes **upward** over time → this means there is an **Upward Trend**',
    '  2. The line has **peaks and dips** that repeat every year → this means there is **Seasonality**',
    '  3. The peaks get **taller** over time → the seasonal amplitude is **expanding** (Multiplicative behavior)'
  ]),
  mc('code', [
    '#we can see that the #passengers is increasing over time, with some seasonality,',
    '#that is, with every year the trend is amplified.'
  ]),

  // --- DECOMPOSITION ---
  mc('code', [
    '#decomposition of the time series - multiplicative model',
    'result = seasonal_decompose(data[[\'#Passengers\']], model = \'multiplicative\', period = 12)',
    'result.plot()',
    'plt.show()'
  ]),
  mc('markdown', [
    '### 📊 How to Read the Decomposition Plot (4 Panels):',
    '1. **Observed (Top):** The raw original data — same as the line plot above.',
    '2. **Trend (Second):** A smooth curve showing the long-term direction. If it goes up, demand is growing.',
    '3. **Seasonal (Third):** A repeating wave pattern. Each wave = 1 year (12 months). Peaks = summer travel. Dips = winter.',
    '4. **Resid (Bottom):** Random noise left over after removing trend and seasonality. Should look random with no clear pattern.'
  ]),

  // --- MANN-KENDALL TEST ---
  mc('code', [
    '#pip install pymannkendall'
  ]),
  mc('code', [
    'import pymannkendall as mk'
  ]),
  mc('code', [
    '#Perform the Mann-Kendall test',
    '#H0: There is no monotonic trend in the series',
    'mk.original_test(data[\'#Passengers\'])'
  ]),
  mc('markdown', [
    '### 📝 How to Read Mann-Kendall Test Result:',
    '- If `trend = \'increasing\'` and `p < 0.05` → statistically confirmed upward trend',
    '- If `trend = \'no trend\'` → no significant monotonic trend exists'
  ]),

  // --- TRAIN TEST SPLIT ---
  mc('code', [
    '#Train test splitting',
    'train_df = data[:int(data.shape[0]*0.7)]',
    'test_df = data[int(data.shape[0]*0.7):]'
  ]),
  mc('code', [
    'train_df.head(5)'
  ]),

  // --- SES ---
  mc('code', [
    '#single exponential smoothing model',
    'from statsmodels.tsa.api import SimpleExpSmoothing',
    'model = SimpleExpSmoothing(train_df)',
    'model_single_fit = model.fit()'
  ]),
  mc('code', [
    'forecast_single = model_single_fit.forecast(len(test_df))',
    'print(forecast_single)'
  ]),
  mc('code', [
    'model_single_fit.params'
  ]),
  mc('code', [
    'plt.plot(data, label ="Original Data")',
    'plt.plot(model_single_fit.fittedvalues, label= "Fitted values")',
    'plt.plot(forecast_single, label = "Forecast")',
    'plt.xlabel("Year")',
    'plt.ylabel("No. of Passengers")',
    'plt.title("Single exponential smoothing")',
    'plt.legend()',
    'plt.show()'
  ]),
  mc('markdown', [
    '### 📊 How to Read the SES Forecast Plot:',
    '- **Blue line** = actual data',
    '- **Orange line** = model\'s fitted values on training data',
    '- **Green line** = forecast into test period',
    '- **Notice:** The green forecast line is **completely flat/horizontal**.',
    '- **Why?** SES only learns the Level (α). It has no trend (β) or seasonality (γ), so it just predicts a constant value.'
  ]),

  // --- HOLT'S (DOUBLE) ---
  mc('code', [
    '#DOUBLE EXPONENETIAL SMOOTHING (Holt\'s model)'
  ]),
  mc('code', [
    'from statsmodels.tsa.api import Holt'
  ]),
  mc('code', [
    'model_double = Holt(train_df)',
    'model_double_fit = model_double.fit()'
  ]),
  mc('code', [
    'forecast_double = model_double_fit.forecast(len(test_df))',
    'print(forecast_double)'
  ]),
  mc('code', [
    'model_double_fit.params'
  ]),
  mc('code', [
    'plt.plot(data, label ="Original Data")',
    'plt.plot(model_double_fit.fittedvalues, label= "Fitted values")',
    'plt.plot(forecast_double, label = "Forecast")',
    'plt.xlabel("Year")',
    'plt.ylabel("No. of Passengers")',
    'plt.title("Double exponential smoothing(Holt\'s model)")',
    'plt.legend()',
    'plt.show()'
  ]),
  mc('markdown', [
    '### 📊 How to Read the Holt\'s Model Forecast Plot:',
    '- **Green forecast line** is now a **straight sloping line** going upward.',
    '- **Why?** Holt\'s model learns Level (α) + Trend (β), so it captures the upward slope.',
    '- **But:** It still misses the seasonal ups and downs — the forecast is a smooth line without waves.'
  ]),

  // --- HOLT-WINTERS ADDITIVE ---
  mc('code', [
    '#TRIPLE EXPONENTIAL SMOOTHING (HOLT_WINTER\'S MODEL)',
    'from statsmodels.tsa.api import ExponentialSmoothing'
  ]),
  mc('code', [
    'model_triple = ExponentialSmoothing(train_df, seasonal_periods = 12, trend = "add", seasonal = "add")',
    '#from the pegel\'s chart, since it looks similar to the additive-additive model,',
    '#therefore both trend and seasonality have been taken as additive',
    'model_triple_fit = model_triple.fit()'
  ]),
  mc('code', [
    'model_triple_fit.params'
  ]),
  mc('code', [
    'forecast_triple = model_triple_fit.forecast(len(test_df))',
    'print(forecast_triple)'
  ]),
  mc('code', [
    'plt.plot(data, label ="Original Data")',
    'plt.plot(model_triple_fit.fittedvalues, label= "Fitted values")',
    'plt.plot(forecast_triple, label = "Forecast")',
    'plt.xlabel("Year")',
    'plt.ylabel("No. of Passengers")',
    'plt.title("Triple exponential smoothing(Holt-winter\'s model) - Additive")',
    'plt.legend()',
    'plt.show()'
  ]),
  mc('markdown', [
    '### 📊 How to Read the Holt-Winters Additive Forecast Plot:',
    '- **Green forecast line** now shows **seasonal waves** — it captures peaks and dips!',
    '- **But:** The wave heights in the forecast remain **constant** (same height peaks).',
    '- In the actual data, peaks are getting **taller** each year — so Additive model **underestimates** later peaks.'
  ]),

  // --- HOLT-WINTERS MULTIPLICATIVE ---
  mc('code', [
    '#using add-multi model',
    'model_triple_mul = ExponentialSmoothing(train_df, seasonal_periods = 12, trend = "add", seasonal = "mul")',
    '',
    'model_triple_fit_mul = model_triple_mul.fit()'
  ]),
  mc('code', [
    'model_triple_fit_mul.params'
  ]),
  mc('code', [
    'forecast_triple_mul = model_triple_fit_mul.forecast(len(test_df))',
    'print(forecast_triple_mul)'
  ]),
  mc('code', [
    'plt.plot(data, label ="Original Data")',
    'plt.plot(model_triple_fit_mul.fittedvalues, label= "Fitted values")',
    'plt.plot(forecast_triple_mul, label = "Forecast")',
    'plt.xlabel("Year")',
    'plt.ylabel("No. of Passengers")',
    'plt.title("Triple exponential smoothing(Holt-winter\'s model) - Multiplicative")',
    'plt.legend()',
    'plt.show()'
  ]),
  mc('markdown', [
    '### 📊 How to Read the Holt-Winters Multiplicative Forecast Plot:',
    '- **Green forecast line** shows seasonal waves that **grow taller** as time progresses.',
    '- This closely matches the actual data where peaks expand proportionally.',
    '- **Best model** for AirPassengers because seasonal amplitude scales with the trend.'
  ]),

  // --- MAPE ---
  mc('code', [
    '#testing for the accuracy of the two models'
  ]),
  mc('code', [
    'from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error'
  ]),
  mc('code', [
    'mape_test_add = mean_absolute_percentage_error(test_df[\'#Passengers\'], forecast_triple)',
    'print("MAPE Test for Test Data:",mape_test_add)'
  ]),
  mc('code', [
    'mape_test_mul = mean_absolute_percentage_error(test_df[\'#Passengers\'], forecast_triple_mul)',
    'print("MAPE Test for Test Data:",mape_test_mul)'
  ]),
  mc('markdown', [
    '### 📝 How to Read MAPE:',
    '- MAPE = Mean Absolute Percentage Error. Lower value = better accuracy.',
    '- Multiplicative model gives lower MAPE → it is the better model for this data.'
  ]),

  // --- STATIONARITY TESTING ---
  mc('code', [
    '#ADF Test',
    '#H0: Series is not stationary, ie,e series has a unit root',
    '#H1: Series is stationary, ie,e series has no unit root'
  ]),
  mc('code', [
    'from statsmodels.tsa.stattools import adfuller ',
    'result = adfuller(data[\'#Passengers\'])',
    '',
    'print("ADF Statistic:", result[0])',
    'print("p-value:",result[1])'
  ]),
  mc('code', [
    '#p-value > 0.05: fail to reject to H0, i.e, series is not stationary'
  ]),
  mc('code', [
    '#KPSS Test',
    '#H0: Series is trend stationary, ie,e series has no unit root',
    '#H1: Series is non-stationary, ie,e series has a unit root'
  ]),
  mc('code', [
    '##It is very specific to trend (KPSS)'
  ]),
  mc('code', [
    'from statsmodels.tsa.stattools import kpss',
    'kp = kpss(data[\'#Passengers\'])',
    'p = kp[1]',
    '',
    'print("p-value for KPSS Test(untransformed) = ",p)'
  ]),
  mc('code', [
    '#p-value = 0.01 < 0.05: Reject HO, ie., series is non stationary'
  ]),
  mc('code', [
    '#Here, Since both the condition is not satisfied, therefore we conclude that the series is NOT stationarity.'
  ]),
  mc('markdown', [
    '### 📝 How to Read ADF & KPSS Results:',
    '- **ADF Test**: If p-value < 0.05 → reject H0 → series IS stationary',
    '- **KPSS Test**: If p-value > 0.05 → fail to reject H0 → series IS stationary',
    '- **Both must agree** for full stationarity. Here both say non-stationary → we need differencing.'
  ]),

  // --- ACF & PACF RAW ---
  mc('code', [
    'from statsmodels.tsa.stattools import acf, pacf',
    'from statsmodels.graphics.tsaplots import plot_acf, plot_pacf',
    '',
    'plt.figure(figsize = (20,5))',
    'plt.grid()',
    'plot_acf(data[\'#Passengers\'], ax = plt.gca(), lags = 30)',
    'plt.show()'
  ]),
  mc('code', [
    'from statsmodels.tsa.stattools import acf, pacf',
    'from statsmodels.graphics.tsaplots import plot_acf, plot_pacf',
    '',
    'plt.figure(figsize = (20,5))',
    'plt.grid()',
    'plot_pacf(data[\'#Passengers\'], ax = plt.gca(), lags = 30)',
    'plt.show()'
  ]),
  mc('markdown', [
    '### 📊 How to Read ACF & PACF Plots:',
    '- **ACF (Autocorrelation Function)**: Shows correlation of the series with its own lagged values.',
    '  - If bars decay **slowly** → series is non-stationary.',
    '  - If bars show **bumps at regular intervals** (e.g., every 12 lags) → seasonality exists.',
    '- **PACF (Partial Autocorrelation Function)**: Shows direct correlation after removing intermediate effects.',
    '  - A sharp cutoff at lag p suggests AR(p) order.',
    '- **Blue shaded region** = 95% confidence interval. Bars outside this region are statistically significant.'
  ]),

  // --- NON-SEASONAL DIFFERENCING ---
  mc('code', [
    '#To make the series stationary we now do differencing'
  ]),
  mc('code', [
    '#NON Seasonal differencing',
    'diff = data[\'#Passengers\'].diff().dropna()',
    'plt.figure(figsize = (14,3))  #making trend data stationary',
    'plt.grid()',
    'plt.plot(diff)',
    'plt.show()'
  ]),
  mc('markdown', [
    '### 📊 How to Read the Non-Seasonal Differenced Plot:',
    '- The series now fluctuates **around zero** → trend has been removed.',
    '- But notice there are still **regular up-down patterns** repeating → seasonality remains.'
  ]),
  mc('code', [
    'from statsmodels.tsa.stattools import adfuller ',
    'result = adfuller(diff)',
    '',
    'print("ADF Statistic:", result[0])',
    'print("p-value:",result[1])'
  ]),
  mc('code', [
    '#Since, p-value < 0.05, reject HO, therfore the stationary is stationary.'
  ]),
  mc('code', [
    '#KPSS Test ',
    'from statsmodels.tsa.stattools import kpss',
    'kp = kpss(diff)',
    'p = kp[1]',
    '',
    'print("p-value for KPSS Test(untransformed) = ",p)'
  ]),
  mc('code', [
    '#p-value < 0.05, do not reject HO, ie., data has become trend stationary'
  ]),
  mc('code', [
    'from statsmodels.tsa.stattools import acf, pacf',
    'from statsmodels.graphics.tsaplots import plot_acf, plot_pacf',
    '',
    'plt.figure(figsize = (20,5))',
    'plt.grid()',
    'plot_acf(diff, ax = plt.gca(), lags = 30)',
    'plt.show()'
  ]),
  mc('code', [
    'from statsmodels.tsa.stattools import acf, pacf',
    'from statsmodels.graphics.tsaplots import plot_acf, plot_pacf',
    '',
    'plt.figure(figsize = (20,5))',
    'plt.grid()',
    'plot_pacf(diff, ax = plt.gca(), lags = 30)',
    'plt.show()'
  ]),
  mc('code', [
    '#according to the plots the series has yet not become stationary.'
  ]),

  // --- SEASONAL DIFFERENCING ---
  mc('code', [
    '#Seasonal differencing',
    'sdiff = data[\'#Passengers\'].diff(periods = 12).dropna()',
    'plt.figure(figsize = (14,3))  #making trend data stationary',
    'plt.grid()',
    'plt.plot(sdiff)',
    'plt.show()'
  ]),
  mc('code', [
    '#ADF test',
    'from statsmodels.tsa.stattools import adfuller ',
    'result = adfuller(sdiff)',
    '',
    'print("ADF Statistic:", result[0])',
    'print("p-value:",result[1])'
  ]),
  mc('code', [
    '#p-value<0.05, reject HO, series is stationary'
  ]),
  mc('code', [
    '#KPSS Test',
    '#KPSS Test ',
    'from statsmodels.tsa.stattools import kpss',
    'kp = kpss(sdiff)',
    'p = kp[1]',
    '',
    'print("p-value for KPSS Test(untransformed) = ",p)'
  ]),
  mc('code', [
    '#p-value < 0.05, reject H0, which means the series is not stationary(trend).'
  ]),
  mc('code', [
    'from statsmodels.tsa.stattools import acf, pacf',
    'from statsmodels.graphics.tsaplots import plot_acf, plot_pacf',
    '',
    'plt.figure(figsize = (20,5))',
    'plt.grid()',
    'plot_acf(sdiff, ax = plt.gca(), lags = 30)',
    'plt.show()'
  ]),
  mc('code', [
    'from statsmodels.tsa.stattools import acf, pacf',
    'from statsmodels.graphics.tsaplots import plot_acf, plot_pacf',
    '',
    'plt.figure(figsize = (20,5))',
    'plt.grid()',
    'plot_pacf(sdiff, ax = plt.gca(), lags = 30)',
    'plt.show()'
  ]),
  mc('code', [
    '#Still not stationary'
  ]),

  // --- COMBINED DIFFERENCING ---
  mc('code', [
    '#SEASONAL AND NON SEASONAL DIFFERNECING',
    'sddiff = sdiff.diff().dropna() #Non seasonal differencing on the seasonal differenced series',
    'plt.figure(figsize = (14,3))',
    'plt.grid()',
    'plt.plot(sddiff)',
    '',
    'plt.show()'
  ]),
  mc('markdown', [
    '### 📊 How to Read the Combined Differenced Plot:',
    '- The series fluctuates **randomly around zero** with no visible pattern.',
    '- Variance looks **constant** throughout → no expanding/contracting oscillations.',
    '- This indicates the series is now **stationary**.'
  ]),
  mc('code', [
    '#ADF test',
    'from statsmodels.tsa.stattools import adfuller ',
    'result = adfuller(sddiff)',
    '',
    'print("ADF Statistic:", result[0])',
    'print("p-value:",result[1])'
  ]),
  mc('code', [
    '#Series is stationary'
  ]),
  mc('code', [
    '#KPSS Test',
    '#KPSS Test ',
    'from statsmodels.tsa.stattools import kpss',
    'kp = kpss(sddiff)',
    'p = kp[1]',
    '',
    'print("p-value for KPSS Test(untransformed) = ",p)'
  ]),
  mc('code', [
    '#Series is trend stationary (p>0.05)'
  ]),
  mc('code', [
    'from statsmodels.tsa.stattools import acf, pacf',
    'from statsmodels.graphics.tsaplots import plot_acf, plot_pacf',
    '',
    'plt.figure(figsize = (20,5))',
    'plt.grid()',
    'plot_acf(sddiff, ax = plt.gca(), lags = 30)',
    'plt.show()'
  ]),
  mc('code', [
    'from statsmodels.tsa.stattools import acf, pacf',
    'from statsmodels.graphics.tsaplots import plot_acf, plot_pacf',
    '',
    'plt.figure(figsize = (20,5))',
    'plt.grid()',
    'plot_pacf(sddiff, ax = plt.gca(), lags = 30)',
    'plt.show()'
  ]),
  mc('markdown', [
    '### 📊 How to Read Final ACF & PACF:',
    '- All bars (except lag 0) fall **inside the blue shaded confidence bounds**.',
    '- This confirms the series has **no significant autocorrelation** left.',
    '- The series is now **fully stationary** and ready for ARIMA / SARIMA modeling.'
  ]),
  mc('code', [
    '#We finally have confirmed that the series is now stationary, via plots as well as tests.'
  ])
];

fs.writeFileSync(path.join(out, 'Practical_1.ipynb'), JSON.stringify(nb(p1), null, 2));
console.log('Practical_1.ipynb written');
