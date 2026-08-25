# Practical 2: CO2 Concentration — Complete Guide

## Dataset
`CO2 Concentration.xls` — Monthly atmospheric CO2 measurements with separate `Year` and `Month` columns.

---

## What This Practical Covers

### Key Difference from Practical 1
- **Date handling**: Raw data has `Year` (1958) and `Month` (3) as separate numeric columns. Must be combined into `"1958-03"` using string concatenation + `zfill(2)` for zero-padding.
- **Additive Decomposition**: Unlike AirPassengers where seasonal amplitude expanded, CO2 seasonal oscillations have **constant height** → use `model='additive'`.

### Step-by-Step Flow
1. **Load data** and combine `Year + Month` → `year_month` datetime column
2. **Drop** raw Year/Month columns after indexing
3. **Line plot** — observe upward trend with constant-height seasonal waves
4. **Additive Decomposition** (`model='additive'`, `period=12`)
5. **Mann-Kendall Test** — confirm upward trend
6. **Train-Test Split** (70/30)
7. **4 Models**: SES, Holt's, Holt-Winters Add, Holt-Winters Mul
8. **MAPE comparison** — Additive ≈ Multiplicative (identical scores because seasonal amplitude is constant)
9. **ADF & KPSS Tests** on raw data

---

## How to Read Each Graph

| Graph | What to Look For |
|-------|-----------------|
| **Line Plot** | Steady upward trend. Seasonal waves have **same height** throughout (unlike AirPassengers) |
| **Additive Decomposition** | Trend = smooth upward curve; Seasonal = uniform annual wave; Resid = very clean minimal noise |
| **SES Forecast** | Flat horizontal line |
| **Holt's Forecast** | Straight upward-sloping line |
| **Holt-Winters Add** | Seasonal forecast waves closely match actual data |
| **Holt-Winters Mul** | Nearly identical to Additive (because seasonal amplitude is constant) |

---

## Key Code: Date Preprocessing
```python
# Combine separate Year and Month columns
df['year_month'] = df['Year'].astype(str) + '-' + df['Month'].astype(str).str.zfill(2)
# zfill(2) pads single-digit months: 3 → "03"

df['year_month'] = pd.to_datetime(df['year_month'])
df = df.set_index('year_month')
df.drop(columns=['Year', 'Month'], inplace=True)
```

---

## Key Finding
> Additive and Multiplicative MAPE scores are **identical** because when seasonal oscillation height does not change over time, both formulations produce the same mathematical result.
