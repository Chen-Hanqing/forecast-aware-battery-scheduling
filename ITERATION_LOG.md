# Iteration log

The chronological history behind `README.md`: what was tried, what broke, what was
found along the way, and why the project ended up with its current shape. The README
states the current methodology and results; this file is the "how we got there."


> Sections 1–14 record what was believed at the time. Several conclusions in §7, §9, §11, §12 and
> §14 were later found to be wrong or overstated; §15 says which and why. The README reports the
> corrected results.

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

## 14. From "τ beats MAE" to "no universal metric"

A single worst-day diagnostic (`scripts/diagnose_tau.py`, scanning per-day τ for LEAR)
led to a much larger reworking of what this project actually claims. Along the way:

- `pipeline._backtest()` was instrumented to record a per-day, per-candidate `daily`
  list (MAE, τ, realised cost, baseline/oracle cost) instead of only 30-day aggregates —
  needed to ask whether τ tracks *daily* economic value, not just the 3-model aggregate
  ranking.
- Decomposing τ into **top-4/bottom-4 extreme-hour recall** (share of the actual
  4 most-expensive/cheapest hours a forecast also puts in its own top/bottom-4) surfaced
  a sharper, asymmetric pattern aggregate τ hides: LEAR finds cheap hours normally
  (recall 0.57) but is dramatically worse at finding expensive ones (0.12 vs. 0.42–0.45
  for the other two candidates) — its weak τ is concentrated specifically in missing
  price spikes, not spread evenly across the ranking.
- Extending that daily breakdown to correlate τ/MAE/recall against realised economic
  value (`scripts/analyze_daily.py`, pooled across 90 day×candidate observations and
  broken out per candidate) found the daily relationship is much messier than the
  aggregate one: no single metric — not even the top-4 recall that looks cleanest in
  aggregate — reliably predicts one model's day-to-day economic outcome, and which
  metric comes closest depends on the model and which economic target (savings vs.
  baseline, or gap to oracle) is asked about. A leave-one-day-out check (excluding
  Jan 15, LEAR's worst day) confirmed none of this was being driven by a single outlier.
- This reframed the project's central claim from "rank correlation beats MAE" (which
  the daily data doesn't cleanly support) to "no single metric — MAE, τ, or extreme-hour
  recall — is a scalar, universal descriptor of forecast quality for a storage decision";
  RQ1/RQ2 and README §5's Discussion were rewritten around this, and a research-quality
  review pass caught several places where the README still overclaimed against this more
  careful picture: "replicating" external papers' results (softened to "an independent
  result consistent with"), a factual contradiction (the calm-regime discussion claiming
  "little theoretical arbitrage value" when the oracle shows 75.5% was avoidable), and a
  headline "60–75%" oracle-avoidable range that, checked against the actual numbers
  (75.5% real-data, 36.6% synthetic, neither near 60%), turned out to have no real
  source and was corrected to the one verified figure.
- Two follow-on experiments closed out remaining open items from README's Limitations:
  the CVaR sensitivity sweep (§4.4) was re-run with the current (fixed) forecaster —
  with real market fundamentals now in the price model, the battery is profitable at
  every CVaR weight tested, unlike the pre-fix version where risk-aversion was the
  difference between a loss and a smaller loss. And a same-window ablation
  (`configs/real_de_no_household_exog.yaml`, `household_price_exog: false`) tested
  whether the household's own load/PV still add anything to price forecasting once
  residual load (the true market fundamental) is available: MAE/τ/recall are unchanged
  within noise, and realised economics are slightly *better* without them — the
  household features aren't earning their complexity once residual load is present.

## 15. Fixing the evaluation, and what it changed

A robustness pass (confidence intervals, sensitivity to k, a value-of-the-stochastic-solution
check) started by re-reading `pipeline._backtest()`. It turned up problems that the earlier
conclusions depended on.

**Leakage in the backtest.** The real data has no export-price column, so the backtest fell back to
`0.75 × actual price` for the export price it fed to every candidate's LP. With a 5 kW battery on a
household drawing under 1 kW, most discharge is exported, so every candidate was told the true price
shape of the day it was being scored on. PV was also the realised value, not a forecast. The fix is
structural: `_plan()` takes only the training history and the candidate's own forecast, so there is
nothing else to leak. A regression test distorts every realised value on the final day and checks
that the schedule does not move.

**Other defects found on the way.**
- Price forecasts and scenario prices were clipped at zero, although German day-ahead prices go
  negative (Jan 15, 2018 does, at night). Now only load is clipped.
- Scenario LPs let the import price move with the scenario while the export price stayed fixed, so
  the scenarios hedged against a move the export leg never felt. Each scenario now exports at its own
  price path. An intermediate run made this visible: the scenario layer looked harmful for gradient
  boosting (−0.020 EUR/day, CI excluding zero) and the effect disappeared once the objective was
  made consistent.
- The model called "LEAR" had no asinh transform, so it is now called Lasso-AR, and a variant with
  the transform (LEAR-style) is a separate candidate.

**Conclusions that did not survive.**
- "No metric predicts daily value, not even τ or recall" (§14) came from pooled correlations that
  mix in day difficulty: volatile days raise both MAE and the gap to the oracle. With each day's
  common level removed (day-demeaned, with block-bootstrap intervals), τ and top-4 recall track cost
  clearly better than MAE in the volatile scenario and modestly better in the calm window.
- "LEAR fails because a linear model cannot follow price spikes" (§7, §9) was a guess. The asinh
  transform, the obvious fix for spikes, changed nothing. Predicted-peak hours showed the actual
  pattern: the Lasso models put the day's maximum at 12–14h on 26 of 30 days, and the real maximum
  is at 6–8h or 16–18h on 29 of 30. The cause is the calendar features (one sin/cos pair, so one
  hump per day). Adding 24 hour dummies to Lasso-AR, and nothing else, moves top-4 recall from 0.12
  to 0.49 and cost by 0.045 EUR/day (CI 0.027–0.063).
- "The household's own load and PV are mildly counterproductive once residual load is present"
  (§14) rested on point estimates. Paired intervals show no detectable effect.
- "Every candidate is profitable in the high-volatility scenario" (§9) was partly a leak artifact.
  Without the leak, two of five candidates lose money against no battery.
- Cost figures moved as well: seasonal naive 2.87 → 2.97 EUR, gradient boosting 3.59 → 3.27,
  Lasso-AR (then "LEAR") 4.53 → 4.70. The oracle and baseline are unchanged.

**What held.** MAE still ranks the models almost backwards in the calm window (ρ = −0.90, n = 5),
seasonal naive is still cheapest on the point estimate, and the recall asymmetry of the plain Lasso
models is real (it is now explained, and fixed by the hour dummies).

**Method notes.**
- Validation forecasts are memoised across backtest origins (consecutive origins share 13 of 14
  validation origins), about 15× faster with identical output (a test compares cached and uncached
  forecasts). A 30-day, five-candidate run takes well under an hour instead of several.
- Runs were paused for about 30 minutes when the laptop lid was closed (`pmset -g log` shows the
  clamshell sleep). CPU time much lower than wall time is the tell; the monitor tool's 30-minute cap
  is unrelated to that and only ends the watcher, not the run.
- Removed as obsolete: `scripts/diagnose_tau.py` and `results/tau_scan_lear.json` (per-day τ is in
  every run's stored records), `scripts/analyze_daily.py` (superseded by `scripts/robustness.py`), and
  the old flat-schema counterfactual results file.
