# Practical 3: Electricity Production — Complete Guide

## Dataset
`Electric_Production.xls` — Monthly industrial electricity production index (`IPG2211A2N` column) with `DATE` column.

---

## What This Practical Covers

1. **Load & Index** — Parse `DATE` column to datetime, set as index
2. **Line Plot** — Observe long-term growth with seasonal peaks (winter heating, summer cooling)
3. **Multiplicative Decomposition** (`period=12`) — Seasonal amplitude expands over time
4. **Mann-Kendall Test** — Confirm upward trend
5. **Train-Test Split** (70/30)
6. **Holt-Winters Multiplicative** forecast + MAPE evaluation
7. **ADF & KPSS Stationarity Tests** on raw and differenced data

---

## How to Read Each Graph

| Graph | What to Look For |
|-------|-----------------|
| **Line Plot** | Long-term growth with clear winter/summer peaks. Some economic cycle dips visible |
| **Decomposition** | Trend shows economic growth; Seasonal shows annual 12-month cycle; Seasonal amplitude expands |
| **Holt-Winters Mul Forecast** | Expanding seasonal waves that match actual future production levels |

---

## Key Observation
- Electricity production data has **expanding seasonal amplitude** → Multiplicative model is the right choice.
- Combined differencing `diff(12).diff()` achieves stationarity.
