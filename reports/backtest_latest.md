# Backtest report (2026-11-01)

Weights: `settings.yaml`. Universe: nasdaqtrader, price >= $2.0, 20-day $ volume >= $5,000,000. Cost 10 bps per pick.

## Current weights, full period

### 5-day hold

260 weekly periods, 2021-11-05 to 2026-10-23, 1300 picks, 5-day hold.

| Metric | Value |
|---|---|
| Hit rate (pick went up) | 51.4% |
| Beat the benchmark | 48.8% |
| Avg pick return | +0.40% |
| Median pick return | +0.13% |
| Avg excess vs benchmark | +0.16% |
| Avg excess vs universe | +0.12% |
| Mean rank IC | 0.006 (t = 0.6, positive 52.7% of weeks) |
| Top minus bottom quintile | -0.06% |
| Equal-weight picks, total return | +143.83% (benchmark +76.95%) |
| CAGR | +18.86% (benchmark +11.70%) |
| Sharpe | 0.83 (benchmark 0.79) |
| Max drawdown | -28.52% (benchmark -16.89%) |

| Slot | Picks | Hit rate | Avg excess |
|---|---|---|---|
| momentum | 1040 | 51.3% | +0.20% |
| sleeper | 260 | 51.5% | +0.01% |

### 10-day hold

259 weekly periods, 2021-11-05 to 2026-10-16, 1295 picks, 10-day hold.

| Metric | Value |
|---|---|
| Hit rate (pick went up) | 56.4% |
| Beat the benchmark | 52.2% |
| Avg pick return | +1.01% |
| Median pick return | +0.84% |
| Avg excess vs benchmark | +0.53% |
| Avg excess vs universe | +0.44% |
| Mean rank IC | 0.007 (t = 0.6, positive 54.8% of weeks) |
| Top minus bottom quintile | -0.04% |

| Slot | Picks | Hit rate | Avg excess |
|---|---|---|---|
| momentum | 1036 | 56.6% | +0.49% |
| sleeper | 259 | 56.0% | +0.67% |

### 21-day hold

256 weekly periods, 2021-11-05 to 2026-09-25, 1280 picks, 21-day hold.

| Metric | Value |
|---|---|
| Hit rate (pick went up) | 56.0% |
| Beat the benchmark | 49.8% |
| Avg pick return | +2.22% |
| Median pick return | +1.25% |
| Avg excess vs benchmark | +1.09% |
| Avg excess vs universe | +1.00% |
| Mean rank IC | 0.014 (t = 1.4, positive 56.6% of weeks) |
| Top minus bottom quintile | +0.14% |

| Slot | Picks | Hit rate | Avg excess |
|---|---|---|---|
| momentum | 1024 | 55.9% | +0.98% |
| sleeper | 256 | 56.6% | +1.56% |

## Walk-forward tuning (out of sample)

Each test window uses weights picked only from data before it.

### Tuned weights, out of sample

156 weekly periods, 2023-11-03 to 2026-10-23, 780 picks, 5-day hold.

| Metric | Value |
|---|---|
| Hit rate (pick went up) | 51.7% |
| Beat the benchmark | 49.6% |
| Avg pick return | +0.58% |
| Median pick return | +0.20% |
| Avg excess vs benchmark | +0.23% |
| Avg excess vs universe | +0.17% |
| Mean rank IC | 0.002 (t = 0.2, positive 50.0% of weeks) |
| Top minus bottom quintile | -0.02% |
| Equal-weight picks, total return | +110.40% (benchmark +65.82%) |
| CAGR | +27.16% (benchmark +17.75%) |
| Sharpe | 0.91 (benchmark 1.22) |
| Max drawdown | -31.39% (benchmark -16.89%) |

| Slot | Picks | Hit rate | Avg excess |
|---|---|---|---|
| momentum | 624 | 51.6% | +0.20% |
| sleeper | 156 | 51.9% | +0.35% |

### Default weights, same windows

156 weekly periods, 2023-11-03 to 2026-10-23, 780 picks, 5-day hold.

| Metric | Value |
|---|---|
| Hit rate (pick went up) | 55.3% |
| Beat the benchmark | 50.3% |
| Avg pick return | +0.83% |
| Median pick return | +0.43% |
| Avg excess vs benchmark | +0.48% |
| Avg excess vs universe | +0.42% |
| Mean rank IC | 0.009 (t = 0.8, positive 51.3% of weeks) |
| Top minus bottom quintile | -0.00% |
| Equal-weight picks, total return | +228.86% (benchmark +65.82%) |
| CAGR | +46.90% (benchmark +17.75%) |
| Sharpe | 1.69 (benchmark 1.22) |
| Max drawdown | -25.32% (benchmark -16.89%) |

| Slot | Picks | Hit rate | Avg excess |
|---|---|---|---|
| momentum | 624 | 55.1% | +0.47% |
| sleeper | 156 | 55.8% | +0.50% |

### Folds

| train_start | test_start | test_end | train_objective | test_ic_tuned | test_ic_default | test_excess_tuned | test_excess_default |
|---|---|---|---|---|---|---|---|
| 2021-11-05 | 2023-11-03 | 2024-04-26 | 0.4799 | -0.0003 | 0.0066 | 0.0105 | 0.0048 |
| 2022-05-06 | 2024-05-03 | 2024-10-25 | 0.7344 | 0.0080 | 0.0083 | -0.0034 | -0.0053 |
| 2022-11-04 | 2024-11-01 | 2025-04-25 | 0.5589 | 0.0019 | 0.0041 | -0.0017 | -0.0036 |
| 2023-05-05 | 2025-05-02 | 2025-10-24 | 1.0254 | 0.0064 | 0.0341 | 0.0043 | 0.0161 |
| 2023-11-03 | 2025-10-31 | 2026-04-24 | 1.3607 | 0.0197 | 0.0182 | 0.0122 | 0.0156 |
| 2024-05-03 | 2026-05-01 | 2026-10-23 | 1.7642 | -0.0240 | -0.0148 | 0.0072 | 0.0018 |

### New weights (average of fold winners)

| rs | mom_12_1 | prox_52w_high | accel | ud_volume | rel_volume | tightness | trend_template | rsi_sweet | breakout |
|---|---|---|---|---|---|---|---|---|---|
| 0.087 | 0.117 | 0.085 | 0.099 | 0.103 | 0.164 | 0.056 | 0.144 | 0.051 | 0.094 |

## Read this before trusting the numbers

- **Survivorship bias.** The universe is today's listed stocks. Companies that were delisted
  or went bankrupt during the test aren't in it, which flatters absolute returns. Rank IC
  and excess vs the universe are less affected than raw returns.
- **Social buzz isn't in the backtest.** Free sources have no history. Every live run saves
  a snapshot to `data/social_history/`; once there are enough weeks, social can be tested too.
- **Costs.** Each pick is charged the configured round-trip cost. Small caps can cost more.
- **The out-of-sample section is the honest one.** In-sample numbers use weights that may
  have been tuned on the same data.
