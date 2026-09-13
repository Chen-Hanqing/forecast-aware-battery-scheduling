"""Pool the per-day, per-candidate records from a completed backtest
(pipeline._backtest()'s "daily" list) into two things the 30-day aggregate
table can't show:

1. A per-candidate table with MAE, tau, top-4/bottom-4 extreme-hour recall,
   and realised cost side by side.
2. Daily-level correlations between forecast quality (tau, MAE, and top-4/
   bottom-4 extreme-hour recall) and realised economic value (savings vs.
   no-battery baseline; gap to oracle), reported two ways: pooled across all
   (day, candidate) observations (n = days * candidates), and separately
   within each candidate (n = days, that model's own day-to-day variation
   only). Pooling mixes between-candidate variation (e.g. LEAR simply has
   different average tau AND different average cost than naive) with
   within-candidate variation (does LEAR's tau on a good day vs. a bad day
   track its cost that day?) — these can point in different directions, so
   report both rather than only the pooled number. Even the within-candidate
   n=30 series are serially correlated (not iid days). Treat all of this as a
   diagnostic, not a hypothesis test. (The aggregate 30-day mean recall
   numbers in README.md §4.1 are a separate, weaker claim — n=3 model-level
   points — from the daily correlations computed here.)

Also supports a leave-one-day-out check (--exclude-date): recomputes the full
aggregate table (MAE, tau, top-4/bottom-4 recall, realised cost, vs. baseline,
share of oracle value captured) for all three candidates with one date
dropped, printed alongside the full 30-day table, so a single outlier day's
influence on every reported number — not just LEAR's tau, checked separately
in README.md §4.1 — can be read off directly.

Usage:
    python -m scripts.analyze_daily --metrics artifacts_real/run_metrics.json
    python -m scripts.analyze_daily --metrics results/run_metrics_real_de.json --exclude-date 2018-01-15
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr

LABELS = {"seasonal_naive": "Seasonal naive", "gradient_boosting": "Gradient boosting", "lear": "LEAR (Lasso)"}


def print_aggregate_table(daily_rows: list[dict]) -> None:
    unique_days = {d["date"]: d for d in daily_rows}.values()
    baseline_total = sum(d["baseline_cost_eur"] for d in unique_days)
    oracle_total = sum(d["oracle_cost_eur"] for d in unique_days)
    print(f"n_days={len(unique_days)}  baseline={baseline_total:.2f} EUR  oracle={oracle_total:.2f} EUR")
    print(f"{'Candidate':<20}{'MAE':>10}{'tau':>8}{'Top4 recall':>13}{'Bottom4 recall':>16}"
          f"{'Cost':>10}{'vs baseline':>13}{'oracle capture':>16}")
    for name, label in LABELS.items():
        rows = [d for d in daily_rows if d["candidate"] == name]
        taus = [d["kendall_tau"] for d in rows if d["kendall_tau"] is not None]
        cost = sum(d["realised_cost_eur"] for d in rows)
        vs_baseline = 100 * (cost / baseline_total - 1)
        oracle_capture = 100 * (baseline_total - cost) / (baseline_total - oracle_total)
        print(f"{label:<20}{np.mean([d['mae_price_eur_kwh'] for d in rows]):>10.4f}{np.mean(taus):>8.3f}"
              f"{np.mean([d['top4_expensive_recall'] for d in rows]):>13.2f}"
              f"{np.mean([d['bottom4_cheap_recall'] for d in rows]):>16.2f}"
              f"{cost:>10.2f}{vs_baseline:>12.1f}%{oracle_capture:>15.1f}%")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", default="artifacts_real/run_metrics.json")
    parser.add_argument("--exclude-date", default=None, help="leave-one-day-out check, e.g. 2018-01-15")
    args = parser.parse_args()

    metrics = json.loads(Path(args.metrics).read_text())
    backtest = metrics["backtest"]
    daily = backtest["daily"]

    print(f"{'Candidate':<20}{'MAE':>10}{'tau':>8}{'Top4 recall':>13}{'Bottom4 recall':>16}{'Mean daily cost':>17}")
    for name, label in LABELS.items():
        c = backtest["candidates"][name]
        print(f"{label:<20}{c['mean_mae_price_eur_kwh']:>10.4f}{c['mean_kendall_tau_price']:>8.3f}"
              f"{c['mean_top4_expensive_recall']:>13.2f}{c['mean_bottom4_cheap_recall']:>16.2f}"
              f"{c['mean_daily_cost_eur']:>17.3f}")

    if args.exclude_date:
        print("\nFull 30-day aggregate (all candidates, recomputed from daily[] for consistency):")
        print_aggregate_table(daily)
        excluded = [d for d in daily if d["date"] != args.exclude_date]
        print(f"\nExcluding {args.exclude_date}:")
        print_aggregate_table(excluded)

    metric_keys = {"tau": "kendall_tau", "MAE": "mae_price_eur_kwh",
                   "top4_recall": "top4_expensive_recall", "bottom4_recall": "bottom4_cheap_recall"}

    def arrays(rows):
        valid = [d for d in rows if d["kendall_tau"] is not None]
        out = {name: np.array([d[key] for d in valid]) for name, key in metric_keys.items()}
        out["savings"] = np.array([d["baseline_cost_eur"] - d["realised_cost_eur"] for d in valid])
        out["gap"] = np.array([d["realised_cost_eur"] - d["oracle_cost_eur"] for d in valid])
        return out

    def print_corrs(a):
        for x_name in metric_keys:
            for y_name, y in [("savings vs. baseline", a["savings"]), ("-gap to oracle", -a["gap"])]:
                r_p, p_p = pearsonr(a[x_name], y)
                r_s, p_s = spearmanr(a[x_name], y)
                print(f"  corr({x_name}, {y_name}): pearson r={r_p:+.3f} (p={p_p:.3f})  spearman r={r_s:+.3f} (p={p_s:.3f})")

    pooled = arrays(daily)
    print(f"\nPooled across candidates: n={len(pooled['tau'])} (days x candidates — mixes between-candidate and "
          f"within-candidate variation, not independent; see docstring)")
    print_corrs(pooled)

    print("\nWithin each candidate only (day-to-day variation for that one model, n=30 each):")
    for name, label in LABELS.items():
        rows = [d for d in daily if d["candidate"] == name]
        print(f" {label}:")
        print_corrs(arrays(rows))


if __name__ == "__main__":
    main()
