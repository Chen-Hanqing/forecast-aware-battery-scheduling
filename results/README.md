# results/

Committed (not gitignored, unlike `artifacts*/`) copies of expensive-to-recompute
outputs — mainly so a ~60–90 minute real-data backtest doesn't have to be re-run just
to look up a number already computed once. `README.md`'s tables are transcribed from
these files.

| File | Produced by | Schema |
|---|---|---|
| `run_metrics_synthetic.json` | `battery-schedule run --config configs/default.yaml` | current: `backtest.candidates.<name>.*` |
| `run_metrics_real_de.json` | `battery-schedule run --config configs/real_de.yaml` | current: `backtest.candidates.<name>.*` (now also `.mean_top4_expensive_recall`/`.mean_bottom4_cheap_recall`) plus a per-day `backtest.daily` list (`{date, candidate, mae_price_eur_kwh, kendall_tau, top4_expensive_recall, bottom4_cheap_recall, realised_cost_eur, baseline_cost_eur, oracle_cost_eur}`); analyzed by `scripts/analyze_daily.py`, see README.md §4.1 |
| `forecast_value_comparison_counterfactual.json` | the now-retired standalone `scripts/forecast_value_comparison.py`, run on `configs/counterfactual.yaml` | **older, flat schema** (`<name>.*` directly, no `backtest` wrapper) — predates the refactor in `ITERATION_LOG.md` §8 that merged this logic into `pipeline._backtest()`. Re-running `battery-schedule run --config configs/counterfactual.yaml` today would produce the current nested schema instead; this file just hasn't been regenerated since. |
| `tau_scan_lear.json` | `scripts/diagnose_tau.py --config configs/real_de.yaml --candidate lear` | per-backtest-day `{origin, date, tau, mae}`, written incrementally (safe to inspect mid-run); worst day (2018-01-15, τ=0.17) is visualized in `docs/images/diagnostic_day_lear_2018-01-15.png`, produced by `scripts/plot_diagnostic_day.py`, see README.md §4.1 |
| `cvar_sensitivity_real_de.json` | `scripts/cvar_sensitivity.py --config configs/real_de.yaml --candidate gradient_boosting` | `{candidate, baseline_no_battery_eur, days, cvar_weight_<w>: {total_eur, vs_baseline_pct}}` for `w` in `[0.0, 0.2, 0.5, 0.8, 1.0]`; plotted by `scripts/plot_cvar_sensitivity.py` into `docs/images/cvar_sensitivity.png`, see README.md §4.4 |

When adding a new expensive result here, prefer the artifact's own natural output
location as the source of truth during a run (`artifacts_real/`, etc. — gitignored,
fine to regenerate) and copy or point the script at `results/` for what should survive
across sessions.
