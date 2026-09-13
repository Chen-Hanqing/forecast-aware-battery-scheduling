# Iteration log

The chronological history behind `README.md`: what was tried, what broke, what was
found along the way, and why the project ended up with its current shape. The README
states the current methodology and results; this file is the "how we got there."

## 1. Starting point: cleaning up an inherited repo

The repo arrived with a working core pipeline (CLI → forecast → scenarios → CVaR LP →
backtest, tested, CI-covered) plus several orphaned experimental files never wired into
it: a LightGBM quantile forecaster, a Gaussian-copula scenario generator, feature
builders, and ENTSO-E/pvlib data connectors — duplicated across files, mixed
English/Chinese comments, undeclared dependencies. These were reorganized into a
`research/` module (kept separate from the tested production path) rather than deleted
outright, on the reasoning that they showed methodological range. This turned out to be
the wrong call — see §9.

## 2. Synthetic baseline

Ran the existing pipeline on synthetic demo data: gradient boosting selected, 35.3%
lower realised cost than no-battery over a 7-day backtest. This was flagged early as
weak evidence — synthetic data (a smooth curve plus noise) is easy to forecast well and
doesn't test much.

## 3. Real data: household load/PV + real prices

Integrated real data: OPSD's `residential4` household (Konstanz, Germany) for load/PV,
joined with real ENTSO-E day-ahead prices for the DE-AT-LU bidding zone. Positioned the
project's narrative around a market-participant framing (a VPP/aggregator that must
commit to a schedule before the day-ahead auction clears, not a household on an
already-published dynamic tariff — see the discussion of why day-ahead price forecasting
matters at all, since a real dynamic-tariff household would receive tomorrow's price as
a published number, not something to forecast).

## 4. First real-data result: the battery lost money

+13.5% vs. no-battery on the first real 30-day backtest (Dec 2017–Jan 2018 window). Not
a bug: real wholesale price spreads were thin relative to battery degradation cost, and
a `cvar_weight` sensitivity sweep (0 → 1) showed risk-aversion could cut the loss
(17.5% → 5.6%) but not eliminate it, and that maximal risk-aversion (weight=1) was worse
than a moderate setting (0.8) — a real, non-monotonic finding kept in README §4.4.

## 5. The oracle comparison changed the diagnosis

A perfect-foresight oracle (same LP, solved against actual realised prices, no
forecasting) showed **60% cost reduction was theoretically available** — the market
wasn't the problem. The gap between that ceiling and what the real forecaster achieved
was a forecasting-quality problem. This sent the project back into `forecast.py`, which
had two real defects:

- **Recursive multi-step forecasting**: each hour's forecast fed its own near-term
  prediction back in as a `lag_1`/`lag_2` feature, compounding error across the
  24-hour horizon.
- **No exogenous market drivers**: price was forecast from its own lags and calendar
  features only, contrary to standard EPF practice (price is driven by net/residual
  demand, not autoregression alone).

## 6. Rewriting the forecaster

Replaced recursion with **direct multi-step forecasting** (one model per horizon step,
trained only on features knowable at the origin — no lag shorter than 24h). Added load
and PV as price's exogenous drivers, each substituted with its own forecast at
inference time (train-on-actual/deploy-on-forecast, the same pattern real EPF models use
for TSO-published forecasts). Added **LEAR** (LASSO-regularized linear autoregression,
one of the two reference models in the epftoolbox EPF benchmark) as a third real
candidate, plus a rolling `lookback_days` training window (both for speed — a full
2+ year training set per origin was impractically slow — and because recent hours are
more representative than multi-year-old ones).

Along the way: an `sklearn` API mismatch (`LassoCV(n_alphas=...)` renamed to `alphas=...`
in the installed version) had to be fixed; a `lightgbm_quantile` fourth candidate was
built but left disabled by default, since `lightgbm` needs `libomp` on macOS and this
machine had no prebuilt bottle (a from-source LLVM build was judged not worth the time).
A stale editable-install pointer (this repo's `.venv` had been created by, and still
pointed at, a different, unrelated clone of the same project) was also found and fixed —
without it, `battery-schedule run` and any bare `python -c "import battery_schedule"`
were silently executing a different, unmodified copy of the code, though `pytest` was
unaffected since it explicitly prepends `src/` to `sys.path`.

## 7. The headline finding

With the fixed forecaster, gradient boosting nearly broke even with no-battery
(previously the worst by a wide margin). Comparing all three candidates independently —
not "whichever the pipeline auto-selects," see §8 — surfaced the project's core result:
**LEAR had the best MAE but the worst realised profit and worst Kendall's τ of the
three.** Verified this matches an independent published finding
([arXiv:2604.12082](https://arxiv.org/abs/2604.12082)), and that a real, more rigorous
prior-art project pursuing the same "what is a forecast worth to a battery" framing
exists ([clearsmog/energy-forecasting-to-dispatch](https://github.com/clearsmog/energy-forecasting-to-dispatch),
verified to actually exist and be substantive after an initial web search failed to
surface it) — useful for calibrating how much rigor this project should and shouldn't
claim.

## 8. Architecture critique: no more auto-selected backtests

`pipeline._backtest()` originally auto-selected one model (by MAE) at every backtest
origin and reported only that winner's economics — meaning `run_metrics.json` could
only ever show "whichever model MAE picked," which is exactly the number §7 says not to
trust as *the* answer. Rebuilt `_backtest()` to evaluate every configured candidate
independently (the same `select_and_forecast()`, called once per candidate with a
single-item candidates list, so no competition happens) plus the oracle, and retired the
now-redundant standalone `forecast_value_comparison.py` / `oracle_comparison.py`
scripts into this single code path. `pipeline.run()`'s *next-day* forecast step still
auto-selects, since exactly one schedule must actually be deployed — only the backtest's
evaluation, not the live decision, stopped collapsing to a single model.

## 9. The counterfactual, and a real optimizer bug

Built a controlled counterfactual — this household's real load/PV pattern, positionally
overlaid on real 2022–23 (European gas crisis era) ENTSO-E prices — to test whether the
"thin spread" finding was a property of the calm 2015–18 market specifically. First
attempt crashed: HiGHS reported some backtest days as infeasible. Diagnosis (manually
constructing a "do nothing" point and checking it against every constraint) found the LP
required each scenario's daily grid cost to be `>= 0` — mathematically ruling out a
household exporting far more solar than it consumes, which is a legitimate profit, not a
cost. Invisible on the calm, low-PV winter data used until then; the counterfactual's
high-PV days hit it immediately. Fixed in `optimise.py`, with a regression test added
(`test_solve_schedule_allows_net_exporter_day`). The counterfactual then showed every
candidate profitable given the ~10x thicker price spread, but LEAR's Kendall's τ
collapsed to ~0.04 (indistinguishable from random) even though its MAE stayed
reasonable — a second, independent confirmation of §7's pattern, and a sharper one.

## 10. Data-provenance corrections

Two separate points of confusion were caught and corrected:

- `research/external_data.py`'s `fetch_physical_pv_profile()` (a pvlib-based PV model,
  never called anywhere in the actual data pipeline) had default coordinates for
  Toulouse, France — a leftover from an early, unused prototype, not anything the real
  German-household analysis used. Confirmed via `grep` that it was never invoked, then
  removed entirely (see §11).
- The README described the real German dataset's date range as "Oct 2015–Feb 2018,"
  which was the household's own raw meter data range, not the actual final merged
  dataset used for training/backtesting — two months (Jan–Feb 2018) of ENTSO-E price
  data had failed to fetch (a transient server-side error) on the first pull, silently
  truncating the inner-joined `history_de_real.csv` to end 2018-01-01. Re-running the
  fetch later succeeded for the full range; the README was corrected to describe
  whichever range the actual file being used covers, not the raw household series'
  own range, and the general lesson (verify the *merged* file, not one input's
  metadata) was applied to future data-pipeline changes.

## 11. Adding real market fundamentals, and cleaning up research/

An external second opinion identified a real weakness: price's exogenous drivers were
household load/PV, but the true day-ahead-price driver is *system-wide* residual load
(actual grid load minus wind and solar generation) — a single household's own numbers
only correlate with that noisily, through shared regional weather. Added
`fetch_entsoe_residual_load` (ENTSO-E actual load + generation-by-type queries,
month-chunked like the existing price fetch to avoid the same offset-pagination bug),
merged `residual_load_mw` into the real-data CSV as an optional column, and extended
`forecast.py` to use it as a third exogenous driver for price when present (backward
compatible: absent for synthetic/counterfactual data, which fall back to load/PV only).

At the same time, the user asked for the whole codebase to be audited for anything
unused or misleading, given the Toulouse-coordinates incident. Of `research/`'s five
files, only two functions (`fetch_entsoe_day_ahead_prices`,
`fetch_entsoe_residual_load`) were ever actually imported by the real pipeline; the
rest (a quantile-regression forecaster, a Gaussian-copula scenario generator, feature
builders, a standalone demo) had never been run since being written in §1 and were
superseded by everything built since. Rather than keep maintaining a "prototype"
label on code some of which had become load-bearing, the two used functions were
extracted into `scripts/entsoe_client.py` and the entire `research/` directory was
deleted — including `fetch_physical_pv_profile` and its Toulouse default. The lesson
from §1 (dead/parallel code paths are worse than no code, especially once some of it
becomes real) was applied here more thoroughly than §1 itself managed to.

A related smaller fix: `fetch_entsoe_day_ahead_prices`/`fetch_entsoe_residual_load`
had generic placeholder default arguments (`start="2024-01-01"`) that were never
actually used anywhere (every real call site passes explicit dates) — the same failure
mode as the Toulouse coordinates, caught by the same kind of question. Removed the
defaults; these parameters are now required.

## 12. Re-running with residual load

Re-fetched the real German dataset (this time all months succeeded — the earlier
Jan–Feb 2018 gap in §10 was transient). The merged file now runs the full household
range (Oct 2015–Feb 2018, 20,359 hours), which shifted the 30-day backtest window from
Dec 2017–Jan 2018 (§4, includes New Year) to Jan 6–Feb 4, 2018 (no holiday disruption).
Re-ran the full per-candidate backtest with residual load included: results in README
§4.1. Because both the forecaster (residual load added) and the backtest window changed
at once, this is *not* a clean ablation of residual load's effect — that requires a
same-window comparison and is listed as an open next step in README §6.

One incidental finding from comparing this window against §4: naive was the best
performer here (calm, non-holiday period) and the worst performer in the original
Dec–Jan window (holiday-disrupted) — "naive is a strong baseline" turned out to be
regime-dependent, not a fixed property of the model, which is itself a small but
genuine finding (README §5).

## 13. Progress-visibility lessons (operational, not scientific)

Several real-data backtests take 60–90+ minutes (every backtest day refits every
candidate from scratch). Recurring operational issues, fixed as they appeared:
`battery-schedule run`'s stdout is block-buffered when redirected to a file, so a
run's `print()` statements can sit invisible until the process exits — fixed by using
`python -u` (unbuffered) and, later, adding explicit per-day progress printing directly
into `pipeline._backtest()` once that became the standard code path (it had none at
first, since the logic was moved in from a script that printed progress, but the
printing wasn't carried over in the move).
