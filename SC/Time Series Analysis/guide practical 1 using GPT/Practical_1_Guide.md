# Practical 1: AirPassengers — Complete Guide

## Dataset
`AirPassengers.xls` — 144 monthly observations of international airline passengers (1949–1960).

---

## What This Practical Covers

### Part A: Exponential Smoothing Forecasting
1. **Load & preprocess** — Read CSV, convert `Month` to datetime, set as index
2. **Visualize** — Line plot showing upward trend + expanding seasonal oscillations
3. **Multiplicative Decomposition** — Split into Observed, Trend, Seasonal, Residual (`period=12`)
4. **Mann-Kendall Test** — Statistically confirm monotonic trend ($H_0$: No trend)
5. **Train-Test Split** — 70% train / 30% test (sequential, never shuffle!)
6. **4 Exponential Smoothing Models:**
   - **SES** (`SimpleExpSmoothing`) — Level only → flat forecast
   - **Holt's Linear** (`Holt`) — Level + Trend → straight sloped line
   - **Holt-Winters Additive** (`ExponentialSmoothing`, `seasonal='add'`) — Level + Trend + constant seasonal waves
   - **Holt-Winters Multiplicative** (`ExponentialSmoothing`, `seasonal='mul'`) — Level + Trend + expanding seasonal waves (**best model**)
7. **MAPE Evaluation** — Multiplicative achieves lowest error (~2.4%)

### Part B: Stationarity & Differencing Pipeline
8. **ADF Test** — $H_0$: Non-stationary. Reject if $p < 0.05$
9. **KPSS Test** — $H_0$: Stationary. Reject if $p < 0.05$ (i.e., stationary if $p > 0.05$)
10. **ACF & PACF Plots** — Visual stationarity diagnostic
11. **3-Stage Differencing:**
    - `diff()` — Removes trend, seasonality remains
    - `diff(12)` — Removes seasonality, trend remains
    - `sdiff.diff()` — Removes both → **fully stationary** (passes ADF + KPSS)

---

## How to Read Each Graph

| Graph | What to Look For |
|-------|-----------------|
| **Line Plot** | Upward trend + seasonal peaks getting taller each year |
| **Decomposition (4 panels)** | Trend panel = smooth upward curve; Seasonal panel = repeating 12-month wave |
| **SES Forecast** | Flat horizontal line (no trend or seasonality captured) |
| **Holt's Forecast** | Straight sloped line (trend only, no seasonal waves) |
| **Holt-Winters Add** | Seasonal waves with constant peak heights |
| **Holt-Winters Mul** | Seasonal waves with expanding peak heights (matches actual data) |
| **ACF Plot** | Slow decay = non-stationary; Bumps every 12 lags = seasonality |
| **PACF Plot** | Sharp cutoff at lag p suggests AR order |
| **Differenced series** | Should fluctuate randomly around zero with constant variance |

---

## Key Statistical Rules (Exam Reference)

| Test | $H_0$ | Decision | Stationary If |
|------|--------|----------|---------------|
| **ADF** | Series has unit root (non-stationary) | Reject $H_0$ if $p < 0.05$ | $p < 0.05$ |
| **KPSS** | Series is trend-stationary | Reject $H_0$ if $p < 0.05$ | $p > 0.05$ |
| **Mann-Kendall** | No monotonic trend | Reject $H_0$ if $p < 0.05$ | N/A |

> **Golden Rule**: Data is fully stationary only when ADF $p < 0.05$ AND KPSS $p > 0.05$.

---

## Stationarity Transformation Summary

| Stage | Code | ADF Pass? | KPSS Pass? | Status |
|-------|------|-----------|------------|--------|
| Raw data | `data['#Passengers']` | ❌ | ❌ | Non-stationary |
| Non-seasonal diff | `.diff()` | ✅ | ❌ | Partially stationary |
| Seasonal diff | `.diff(12)` | ✅ | ❌ | Partially stationary |
| Combined diff | `.diff(12).diff()` | ✅ | ✅ | **Fully stationary** |
