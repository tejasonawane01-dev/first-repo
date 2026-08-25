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
// PRACTICAL 2: CO2 Concentration
// ============================================================================
const p2 = [
  mc('markdown', [
    '# Practical 2: Time Series Analysis & Forecasting',
    '## Dataset: CO2 Concentration.xls',
    '## Objective: Handle multi-column dates, perform additive decomposition, exponential smoothing, and stationarity testing'
  ]),
  mc('code', [
    'import numpy as np',
    'import pandas as pd',
    'import seaborn as sns',
    'import matplotlib.pyplot as plt',
    'import os',
    'from statsmodels.tsa.seasonal import seasonal_decompose'
  ]),
  mc('code', [
    'df = pd.read_csv("CO2 Concentration.xls")'
  ]),
  mc('code', [
    'df.head(5)'
  ]),
  mc('code', [
    'df.shape'
  ]),
  mc('code', [
    'df[\'year_month\'] = df[\'Year\'].astype(str) + \'-\' + df[\'Month\'].astype(str).str.zfill(2)',
    'df.head(5)'
  ]),
  mc('code', [
    '#convert to datetime format',
    'df[\'year_month\'] = pd.to_datetime(df[\'year_month\'])',
    'df = df.set_index(\'year_month\')'
  ]),
  mc('code', [
    'df.head(5)'
  ]),
  mc('code', [
    'df.drop(columns = [\'Year\',\'Month\'], inplace = True)'
  ]),
  mc('code', [
    'df.head(5)'
  ]),
  mc('code', [
    'sns.lineplot(df)'
  ]),
  mc('markdown', [
    '### 📊 How to Read This Graph:',
    '- CO2 concentration rises steadily from ~315 ppm to 390+ ppm → **strong upward trend**.',
    '- Seasonal peaks and dips repeat every year with **constant height** → **Additive seasonality**.',
    '- Unlike AirPassengers where peaks grew taller, here the wave height stays the same throughout.'
  ]),
  mc('code', [
    '#the model is additive model, because the pattern is same throughout'
  ]),
  mc('code', [
    '#decomposition of the time series - additive model',
    'result = seasonal_decompose(df[[\'CO2 Concentration\']], model = \'additive\', period = 12)',
    'result.plot()',
    'plt.show()'
  ]),
  mc('markdown', [
    '### 📊 How to Read This Decomposition:',
    '- **Observed**: Raw CO2 data.',
    '- **Trend**: Smooth upward curve showing atmospheric CO2 accumulation over decades.',
    '- **Seasonal**: Uniform annual wave pattern (constant height). Peak = May, Dip = October.',
    '- **Resid**: Small random noise — very clean decomposition.'
  ]),
  mc('code', [
    'import pymannkendall as mk'
  ]),
  mc('code', [
    '#Perform the Mann-Kendall test',
    '#H0: There is no monotonic trend in the series',
    'mk.original_test(df[\'CO2 Concentration\'])'
  ]),
  mc('code', [
    '#Train test split',
    '#Train test splitting',
    'train_df = df[:int(df.shape[0]*0.7)]',
    'test_df = df[int(df.shape[0]*0.7):]'
  ]),
  mc('code', [
    'train_df.head(5)'
  ]),
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
    'plt.plot(df, label ="Original Data")',
    'plt.plot(model_single_fit.fittedvalues, label= "Fitted values")',
    'plt.plot(forecast_single, label = "Forecast")',
    'plt.xlabel("Year")',
    'plt.ylabel("CO2 Concentration")',
    'plt.title("Single exponential smoothing")',
    'plt.legend()',
    'plt.show()'
  ]),
  mc('code', [
    '#DOUBLE EXPONENETIAL SMOOTHING (Holt\'s model)',
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
    'plt.plot(df, label ="Original Data")',
    'plt.plot(model_double_fit.fittedvalues, label= "Fitted values")',
    'plt.plot(forecast_double, label = "Forecast")',
    'plt.xlabel("Year")',
    'plt.ylabel("CO2 concentration")',
    'plt.title("Double exponential smoothing(Holt\'s model)")',
    'plt.legend()',
    'plt.show()'
  ]),
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
    'plt.plot(df, label ="Original Data")',
    'plt.plot(model_triple_fit.fittedvalues, label= "Fitted values")',
    'plt.plot(forecast_triple, label = "Forecast")',
    'plt.xlabel("Year")',
    'plt.ylabel("CO2 Concentration")',
    'plt.title("Triple exponential smoothing (Holt-winter\'s model)")',
    'plt.legend()',
    'plt.show()'
  ]),
  mc('code', [
    '#using add-multi model',
    'model_triple_mul = ExponentialSmoothing(train_df, seasonal_periods = 12, trend = "add", seasonal = "mul")',
    '',
    'model_triple_fit_mul = model_triple_mul.fit()'
  ]),
  mc('code', [
    'forecast_triple_mul = model_triple_fit_mul.forecast(len(test_df))',
    'print(forecast_triple_mul)'
  ]),
  mc('code', [
    'plt.plot(df, label ="Original Data")',
    'plt.plot(model_triple_fit_mul.fittedvalues, label= "Fitted values")',
    'plt.plot(forecast_triple_mul, label = "Forecast")',
    'plt.xlabel("Year")',
    'plt.ylabel("CO2 Concentration")',
    'plt.title("Triple exponential smoothing (Holt-winter\'s model)")',
    'plt.legend()',
    'plt.show()'
  ]),
  mc('code', [
    '#testing for the accuracy of the two models',
    'from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error'
  ]),
  mc('code', [
    'mape_test_add = mean_absolute_percentage_error(test_df[\'CO2 Concentration\'], forecast_triple)',
    'print("MAPE Test for Test Data:",mape_test_add)'
  ]),
  mc('code', [
    'mape_test_mul = mean_absolute_percentage_error(test_df[\'CO2 Concentration\'], forecast_triple_mul)',
    'print("MAPE Test for Test Data:",mape_test_mul)'
  ]),
  mc('code', [
    '#No difference in both methodologies'
  ]),
  mc('markdown', [
    '### 📝 Why are Additive and Multiplicative MAPE scores identical?',
    '- Because CO2 seasonal oscillations have **constant height** throughout.',
    '- When seasonal amplitude does NOT expand, both additive and multiplicative formulas produce the same forecast.'
  ]),
  mc('code', [
    '#ADF Test',
    '#H0: Series is not stationary, ie,e series has a unit root',
    '#H1: Series is stationary, ie,e series has no unit root'
  ]),
  mc('code', [
    'from statsmodels.tsa.stattools import adfuller ',
    'result = adfuller(df[\'CO2 Concentration\'])',
    '',
    'print("ADF Statistic:", result[0])',
    'print("p-value:",result[1])'
  ]),
  mc('code', [
    '#p-value > 0.05 : fail to reject to H0, i.e, series is not stationary'
  ]),
  mc('code', [
    '#KPSS Test',
    '#H0: Series is trend stationary, ie,e series has no unit root',
    '#H1: Series is non-stationary, ie,e series has a unit root',
    '',
    '##It is very specific to trend (KPSS)'
  ]),
  mc('code', [
    'from statsmodels.tsa.stattools import kpss',
    'kp = kpss(df[\'CO2 Concentration\'])',
    'p = kp[1]',
    '',
    'print("p-value for KPSS Test(untransformed) = ",p)'
  ])
];

// ============================================================================
// PRACTICAL 3: Electric Production
// ============================================================================
const p3 = [
  mc('markdown', [
    '# Practical 3: Time Series Analysis & Forecasting',
    '## Dataset: Electric_Production.xls',
    '## Objective: Perform multiplicative decomposition, Holt-Winters forecasting, and stationarity analysis on electricity production data'
  ]),
  mc('code', [
    'import pandas as pd',
    'import numpy as np ',
    'import os',
    'import matplotlib.pyplot as plt',
    'from statsmodels.tsa.seasonal import seasonal_decompose',
    'import seaborn as sns'
  ]),
  mc('code', [
    'df = pd.read_csv(\'Electric_Production.xls\')',
    'df.head()'
  ]),
  mc('code', [
    'df.shape'
  ]),
  mc('code', [
    '# convert into date time',
    'df[\'DATE\'] = pd.to_datetime(df[\'DATE\'])',
    'df = df.set_index(\'DATE\')',
    'df.head()'
  ]),
  mc('code', [
    'sns.lineplot(df)',
    'plt.ylabel(\'Electricity_Production\')'
  ]),
  mc('markdown', [
    '### 📊 How to Read This Graph:',
    '- Electricity production shows a **long-term upward trend** with occasional dips (economic cycles).',
    '- There are clear **seasonal peaks** every winter (heating) and summer (cooling).',
    '- The seasonal amplitude **expands** over time → **Multiplicative model** is appropriate.'
  ]),
  mc('code', [
    'result = seasonal_decompose(df[[\'IPG2211A2N\']],',
    '                            model = \'multiplicative\',',
    '                            period = 12)',
    'result.plot()',
    'plt.show()'
  ]),
  mc('markdown', [
    '### 📊 How to Read This Decomposition:',
    '- **Observed**: Raw electricity production index.',
    '- **Trend**: Long-term economic growth in electricity demand.',
    '- **Seasonal**: Annual 12-month cycle showing winter/summer demand peaks.',
    '- **Resid**: Random unexplained fluctuations.'
  ]),
  mc('code', [
    'import pymannkendall as mk',
    'mk.original_test(df[\'IPG2211A2N\'])'
  ]),
  mc('code', [
    '#Train test splitting',
    'train_df = df[:int(df.shape[0]*0.7)]',
    'test_df = df[int(df.shape[0]*0.7):]'
  ]),
  mc('code', [
    'from statsmodels.tsa.api import ExponentialSmoothing',
    '',
    'model_triple_mul = ExponentialSmoothing(train_df[\'IPG2211A2N\'], seasonal_periods = 12, trend = "add", seasonal = "mul")',
    'model_triple_fit_mul = model_triple_mul.fit()'
  ]),
  mc('code', [
    'forecast_triple_mul = model_triple_fit_mul.forecast(len(test_df))',
    'print(forecast_triple_mul)'
  ]),
  mc('code', [
    'plt.plot(df[\'IPG2211A2N\'], label ="Original Data")',
    'plt.plot(model_triple_fit_mul.fittedvalues, label= "Fitted values")',
    'plt.plot(forecast_triple_mul, label = "Forecast")',
    'plt.title("Holt-Winters Multiplicative - Electricity Production")',
    'plt.legend()',
    'plt.show()'
  ]),
  mc('code', [
    'from sklearn.metrics import mean_absolute_percentage_error',
    'mape = mean_absolute_percentage_error(test_df[\'IPG2211A2N\'], forecast_triple_mul)',
    'print("MAPE:", mape)'
  ]),
  mc('code', [
    'from statsmodels.tsa.stattools import adfuller, kpss',
    '',
    'print("Raw ADF p-value:", adfuller(df[\'IPG2211A2N\'])[1])',
    'print("Raw KPSS p-value:", kpss(df[\'IPG2211A2N\'])[1])'
  ]),
  mc('code', [
    '#Combined Seasonal and Non-Seasonal Differencing',
    'sddiff = df[\'IPG2211A2N\'].diff(12).diff().dropna()',
    '',
    'print("Differenced ADF p-value:", adfuller(sddiff)[1])',
    'print("Differenced KPSS p-value:", kpss(sddiff)[1])'
  ])
];

// ============================================================================
// PRACTICAL 4: Yearly Revenue
// ============================================================================
const p4 = [
  mc('markdown', [
    '# Practical 4: Time Series Analysis',
    '## Dataset: Yearly_revenue.xls',
    '## Objective: Handle annual (non-seasonal) data, perform integer indexing, decomposition with period=1, and Holt\'s Linear model'
  ]),
  mc('code', [
    'import pandas as pd',
    'import numpy as np ',
    'import os',
    'import matplotlib.pyplot as plt',
    'from statsmodels.tsa.seasonal import seasonal_decompose',
    'import seaborn as sns'
  ]),
  mc('code', [
    'df = pd.read_csv(r\'Yearly_revenue.xls\')',
    'df.head()'
  ]),
  mc('code', [
    'df.shape'
  ]),
  mc('code', [
    'df[\'Year\'] = range(1, len(df) + 1)',
    'df = df.set_index("Year") '
  ]),
  mc('code', [
    'sns.lineplot(data=df)'
  ]),
  mc('markdown', [
    '### 📊 How to Read This Graph:',
    '- Shows year-over-year revenue growth.',
    '- Since this is **annual** data (one point per year), there are **no seasonal cycles**.',
    '- Only a **trend** component exists.'
  ]),
  mc('code', [
    'result = seasonal_decompose(df, model=\'multiplicative\', period=1)',
    'result.plot()',
    'plt.show()'
  ]),
  mc('markdown', [
    '### 📝 Why period=1?',
    '- With annual data there are no sub-cycles (no monthly or quarterly seasonality).',
    '- `period=1` tells decompose there is no seasonal component to extract.'
  ]),
  mc('code', [
    'from statsmodels.tsa.api import Holt',
    'target_col = df.columns[0]',
    '',
    'model_double = Holt(df[target_col])',
    'model_double_fit = model_double.fit()',
    'print(model_double_fit.params)'
  ]),
  mc('code', [
    'from statsmodels.tsa.stattools import adfuller, kpss',
    '',
    'diff_rev = df[target_col].diff().dropna()',
    'print("First Diff ADF p-value:", adfuller(diff_rev)[1])',
    'print("First Diff KPSS p-value:", kpss(diff_rev)[1])'
  ])
];

// ============================================================================
// PRACTICAL 5: Catfish Sales ARIMA
// ============================================================================
const p5 = [
  mc('markdown', [
    '# Practical 5: Box-Jenkins ARIMA Modeling',
    '## Dataset: catfish.xls',
    '## Objective: Perform stationarity testing, determine ARIMA orders (p,d,q) from ACF/PACF plots, fit and forecast using ARIMA model'
  ]),
  mc('code', [
    'import numpy as np',
    'import pandas as pd',
    'import seaborn as sns',
    'import matplotlib.pyplot as plt',
    'from statsmodels.tsa.stattools import adfuller, kpss',
    'from statsmodels.graphics.tsaplots import plot_acf, plot_pacf'
  ]),
  mc('code', [
    'df = pd.read_csv("catfish.xls")'
  ]),
  mc('code', [
    'df.head(5)'
  ]),
  mc('code', [
    'df.shape'
  ]),
  mc('code', [
    'df["Date"] = pd.to_datetime(df["Date"])',
    'df = df.set_index("Date")'
  ]),
  mc('code', [
    'df.head(5)'
  ]),
  mc('code', [
    'sns.lineplot(df)',
    'plt.ylabel("Catfish Sales")'
  ]),
  mc('markdown', [
    '### 📊 How to Read This Graph:',
    '- Monthly catfish sales showing trend and some volatility.',
    '- We need to check if this data is stationary before applying ARIMA.'
  ]),
  mc('code', [
    '#ADF and KPSS Test on Raw Data',
    'print("ADF Statistic:", adfuller(df["Total"])[0])',
    'print("ADF p-value:", adfuller(df["Total"])[1])',
    'print("KPSS p-value:", kpss(df["Total"])[1])'
  ]),
  mc('code', [
    '#First Differencing for Stationarity (d=1)',
    'diff_catfish = df["Total"].diff().dropna()',
    '',
    'print("Differenced ADF p-value:", adfuller(diff_catfish)[1])',
    'print("Differenced KPSS p-value:", kpss(diff_catfish)[1])'
  ]),
  mc('code', [
    '#ACF Plot - used to determine MA order (q)',
    'plt.figure(figsize = (20,5))',
    'plt.grid()',
    'plot_acf(diff_catfish, ax = plt.gca(), lags = 30)',
    'plt.show()'
  ]),
  mc('code', [
    '#PACF Plot - used to determine AR order (p)',
    'plt.figure(figsize = (20,5))',
    'plt.grid()',
    'plot_pacf(diff_catfish, ax = plt.gca(), lags = 30)',
    'plt.show()'
  ]),
  mc('markdown', [
    '### 📊 How to Read ACF & PACF for ARIMA Order Selection:',
    '- **PACF**: Look at where bars first cut into the blue shaded region. That lag = AR order **p**.',
    '- **ACF**: Look at where bars first cut into the blue shaded region. That lag = MA order **q**.',
    '- **d** = number of times we differenced the data (here d=1).'
  ]),
  mc('code', [
    '#Train Test Split (80% Train, 20% Test)',
    'train_df = df[:int(df.shape[0]*0.8)]',
    'test_df = df[int(df.shape[0]*0.8):]'
  ]),
  mc('code', [
    '#Fit ARIMA(1, 1, 1) Model',
    'from statsmodels.tsa.arima.model import ARIMA',
    '',
    'model_arima = ARIMA(train_df["Total"], order=(1, 1, 1))',
    'model_arima_fit = model_arima.fit()',
    'print(model_arima_fit.summary())'
  ]),
  mc('markdown', [
    '### 📝 How to Read ARIMA Summary Table:',
    '- **coef**: The learned AR and MA coefficient values.',
    '- **P>|z|**: The p-value for each coefficient. If p < 0.05, that coefficient is statistically significant.',
    '- **AIC**: Akaike Information Criterion. Lower AIC = better model.'
  ]),
  mc('code', [
    '#Forecast and MAPE Evaluation',
    'from sklearn.metrics import mean_absolute_percentage_error',
    '',
    'forecast_arima = model_arima_fit.forecast(len(test_df))',
    'mape = mean_absolute_percentage_error(test_df["Total"], forecast_arima)',
    'print("ARIMA Forecast MAPE:", mape)'
  ]),
  mc('code', [
    '#Plot Actual vs Forecast',
    'plt.plot(df["Total"], label="Original Data")',
    'plt.plot(forecast_arima, label="ARIMA Forecast")',
    'plt.legend()',
    'plt.title("Catfish Sales ARIMA Forecast")',
    'plt.show()'
  ]),
  mc('markdown', [
    '### 📊 How to Read the ARIMA Forecast Plot:',
    '- **Blue line** = full original data (train + test).',
    '- **Orange line** = ARIMA model forecast for the test period.',
    '- Compare how closely the orange line follows the blue line in the test region.'
  ])
];

// Write all notebooks
fs.writeFileSync(path.join(out, 'Practical_2.ipynb'), JSON.stringify(nb(p2), null, 2));
fs.writeFileSync(path.join(out, 'Practical_3.ipynb'), JSON.stringify(nb(p3), null, 2));
fs.writeFileSync(path.join(out, 'Practical_4.ipynb'), JSON.stringify(nb(p4), null, 2));
fs.writeFileSync(path.join(out, 'Practical_5.ipynb'), JSON.stringify(nb(p5), null, 2));

console.log('Practical_2 through Practical_5 .ipynb written!');
