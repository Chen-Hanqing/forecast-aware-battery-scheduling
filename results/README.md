# results/

Committed (not gitignored, unlike `artifacts*/`) copies of expensive-to-recompute outputs, split
into the two studies described in the top-level [README](../README.md) and
[docs/household_study.md](../docs/household_study.md).

## `market/` — the three-market, six-year merchant-battery study

| File | Produced by | Contents |
|---|---|---|
| `run_metrics_market_{de_lu,nl,fr}.json` | `battery-schedule run --config configs/market_<market>.yaml` | Main pass: naive, gradient boosting, Lasso-AR, Lasso-AR + hour dummies (France's config also includes the two aligned candidates in this one pass). Per-day records: metrics, forecast/actual price vectors, charge/discharge schedule, realised cost. |
| `run_metrics_market_{de_lu,nl}_aligned.json` | `battery-schedule run --config configs/market_<market>_aligned.yaml` | Second pass: naive (repeated, as a consistency check against the main pass) plus gradient-boosting and Lasso-AR with lags aligned to the target hour. |
| `report_{de_lu,nl,fr}.txt`, `market_report_{de_lu,nl,fr}.json` | `python -m scripts.market_report --metrics <run_metrics files> --out <json>` | Revenue and risk by candidate, by calendar year, paired daily-revenue differences (block-bootstrap CI), forecast quality (MAE/τ/recall), model-level and day-demeaned correlations with revenue, peak-hour placement. |
| `oracle_net_{de_lu,nl,fr}.json` | `scripts.market_report.oracle_net_revenue` (called automatically) | Perfect-foresight net-of-degradation revenue per day, re-solved from the stored actual prices and checked against the stored gross oracle revenue. |

`scripts/run_market_study.sh` runs the aligned passes, merges each market's two files, and
produces the reports and `docs/images/market_*.png` charts in one go.

## `household/` — the 13.5 kWh / 5 kW household-scale study

| File | Produced by | Contents |
|---|---|---|
| `run_metrics_real_de.json` | `battery-schedule run --config configs/real_de.yaml` | Aggregates per candidate, oracle and baseline, plus a per-day `backtest.daily` list: metrics, forecast and actual price vectors, charge/discharge schedule, realised cost of the CVaR, deterministic and risk-neutral schedules. |
| `run_metrics_real_de_no_household_exog.json` | `battery-schedule run --config configs/real_de_no_household_exog.yaml` | Same schema; household load/PV dropped from price inputs. |
| `run_metrics_counterfactual.json` | `battery-schedule run --config configs/counterfactual.yaml` | Same schema; constructed 2022-23 scenario. |
| `run_metrics_synthetic.json` | `battery-schedule run --config configs/default.yaml` | Same schema; 7-day synthetic sanity check. |
| `robustness_real_de.json`, `robustness_counterfactual.json`, `robustness_real_de_no_household_exog.json` | `python -m scripts.robustness --metrics <run_metrics file> --out <file>` | Block-bootstrap intervals, model-level and day-demeaned correlations, k-sensitivity, stochastic-layer split, forecast sharpness and cycling. The real-data file also holds the paired comparison against the ablation run (`--vs`). |
| `cvar_sensitivity_real_de.json` | `python -m scripts.cvar_sensitivity --config configs/real_de.yaml --candidate gradient_boosting` | Total, worst-day and worst-10%-of-days cost, plus the daily costs, for CVaR weights 0, 0.2, 0.5, 0.8, 1. |

When adding a new expensive result here, prefer the artifact's own natural output location as the
source of truth during a run (`artifacts_market_*/`, `artifacts_real/`, etc. — gitignored, fine
to regenerate) and copy or point the script at `results/market/` or `results/household/` for what
should survive across sessions.
