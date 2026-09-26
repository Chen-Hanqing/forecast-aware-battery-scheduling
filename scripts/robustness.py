"""Uncertainty and sensitivity checks on a completed backtest's per-day records
(pipeline._backtest()'s "daily" list), so the headline table isn't read as more
certain than 30 days of data can support.

1. The stochastic layer, split in two so they aren't confused: (a) the value of modelling
   uncertainty at all (deterministic point-forecast schedule vs. the same scenarios with
   cvar_weight=0) and (b) what CVaR risk aversion then adds or costs (cvar_weight=0 vs. the
   configured weight). Both are ex-post, single-realisation differences in realised cost, not
   textbook expected-value VSS.
2. Paired cost differences between candidates, day by day, with a circular
   moving-block bootstrap CI. Days are serially correlated, so iid resampling
   would give intervals that are too narrow.
3. Sensitivity of top-k / bottom-k extreme-hour recall to k. The battery's full-power
   duration (capacity / power) is a natural k; k=4 in the README is a convention.
4. Block-bootstrap CIs for the pooled daily correlations (resampling whole days,
   keeping every candidate's row for a resampled day together) and for the
   difference between two metrics' correlations with the same economic target.
5. The same correlations after day-demeaning across candidates. Raw pooled correlations mix in
   how hard the day was for everyone (volatile days raise both MAE and the gap to the oracle);
   demeaning asks whether, on the same day, the candidate with the better metric cost less.

All of it is descriptive: 30 days, a handful of candidates. It is there to show how
wide the uncertainty is, not to manufacture significance.

Usage:
    python -m scripts.robustness --metrics results/household/run_metrics_real_de.json --out results/household/robustness_real_de.json
    python -m scripts.robustness --metrics results/household/run_metrics_real_de.json \\
        --vs results/household/run_metrics_real_de_no_household_exog.json   # paired ablation comparison
"""
from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from scripts.style import LABELS

K_VALUES = (2, 3, 4, 5, 6)


def block_indices(n_days: int, block: int, rng: np.random.Generator) -> np.ndarray:
    """One circular moving-block bootstrap resample of day indices."""
    n_blocks = int(np.ceil(n_days / block))
    starts = rng.integers(0, n_days, size=n_blocks)
    idx = (starts[:, None] + np.arange(block)[None, :]) % n_days
    return idx.ravel()[:n_days]


def bootstrap_ci(stat, n_days: int, block: int, n_boot: int, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    draws = [stat(block_indices(n_days, block, rng)) for _ in range(n_boot)]
    lo, hi = np.nanpercentile(draws, [2.5, 97.5])
    return float(lo), float(hi)


def recall_k(forecast: np.ndarray, actual: np.ndarray, k: int) -> tuple[float, float]:
    top = len(set(np.argsort(actual)[-k:]) & set(np.argsort(forecast)[-k:])) / k
    bottom = len(set(np.argsort(actual)[:k]) & set(np.argsort(forecast)[:k])) / k
    return top, bottom


def load_days(metrics_path: str) -> tuple[list[str], dict[str, dict[str, np.ndarray]], dict[str, np.ndarray]]:
    daily = json.loads(Path(metrics_path).read_text())["backtest"]["daily"]
    dates = sorted({d["date"] for d in daily})
    per: dict[str, dict[str, np.ndarray]] = {}
    for name in LABELS:
        rows = {d["date"]: d for d in daily if d["candidate"] == name}
        if len(rows) != len(dates):
            continue
        ordered = [rows[t] for t in dates]
        per[name] = {
            "cost": np.array([r["realised_cost_eur"] for r in ordered]),
            "cost_det": np.array([r["realised_cost_deterministic_eur"] for r in ordered]),
            "cost_rn": np.array([r["realised_cost_risk_neutral_eur"] for r in ordered]),
            "tau": np.array([np.nan if r["kendall_tau"] is None else r["kendall_tau"] for r in ordered]),
            "neg_mae": -np.array([r["mae_price_eur_kwh"] for r in ordered]),  # higher = more accurate
            "top4": np.array([r["top4_expensive_recall"] for r in ordered]),
            "forecast": np.array([r["forecast_price_eur_kwh"] for r in ordered]),
            "actual": np.array([r["actual_price_eur_kwh"] for r in ordered]),
            "charge": np.array([r["charge_kw"] for r in ordered]),
        }
    ref = {d["date"]: d for d in daily}
    shared = {
        "baseline": np.array([ref[t]["baseline_cost_eur"] for t in dates]),
        "oracle": np.array([ref[t]["oracle_cost_eur"] for t in dates]),
    }
    return dates, per, shared


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", default="artifacts_real/run_metrics.json")
    parser.add_argument("--block", type=int, default=5, help="bootstrap block length in days")
    parser.add_argument("--n-boot", type=int, default=5000)
    parser.add_argument("--battery-hours", type=float, default=2.7, help="capacity / power, for reference")
    parser.add_argument("--vs", default=None, help="a second metrics file (e.g. an ablation run) to compare against, candidate by candidate")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    dates, per, shared = load_days(args.metrics)
    n = len(dates)
    names = list(per)
    out: dict = {"days": n, "block_days": args.block, "n_boot": args.n_boot}
    print(f"{n} days, circular block bootstrap (block={args.block} days, B={args.n_boot}); 95% intervals\n")

    print("0. Model-level rank correlation with 30-day realised cost (n = number of candidates; descriptive only)")
    totals = {c: float(per[c]["cost"].sum()) for c in names}
    model_level = {
        "mean_neg_mae": {c: float(per[c]["neg_mae"].mean()) for c in names},
        "mean_tau": {c: float(np.nanmean(per[c]["tau"])) for c in names},
        "mean_top4_recall": {c: float(per[c]["top4"].mean()) for c in names},
    }
    out["model_level_spearman_with_neg_cost"] = {}
    for key, vals in model_level.items():
        rho = float(spearmanr([vals[c] for c in names], [-totals[c] for c in names]).statistic)
        out["model_level_spearman_with_neg_cost"][key] = rho
        print(f"   {key:<18} Spearman rho = {rho:+.2f}   (n={len(names)}; higher metric should mean lower cost)")
    print()
    print("1. Stochastic layer, split in two (EUR/day, positive = that step lowered realised cost)")
    print("   (a) value of modelling uncertainty  = deterministic cost - risk-neutral scenario cost")
    print("   (b) value of CVaR risk aversion     = risk-neutral scenario cost - CVaR-weighted scenario cost")
    out["stochastic_layer"] = {}
    for name in names:
        row = {}
        for key, d in (("modelling_uncertainty", per[name]["cost_det"] - per[name]["cost_rn"]),
                       ("cvar_risk_aversion", per[name]["cost_rn"] - per[name]["cost"])):
            lo, hi = bootstrap_ci(lambda i, d=d: d[i].mean(), n, args.block, args.n_boot, 1)
            row[key] = {"mean_eur_per_day": float(d.mean()), "ci95": [lo, hi], "share_days_positive": float((d > 1e-9).mean())}
        out["stochastic_layer"][name] = row
        a, b = row["modelling_uncertainty"], row["cvar_risk_aversion"]
        print(f"   {LABELS[name]:<20} (a) {a['mean_eur_per_day']:+.4f} [{a['ci95'][0]:+.4f}, {a['ci95'][1]:+.4f}]"
              f"   (b) {b['mean_eur_per_day']:+.4f} [{b['ci95'][0]:+.4f}, {b['ci95'][1]:+.4f}]")

    print("\n2. Paired daily cost differences a - b (EUR/day; <0 = a cheaper)")
    out["paired_cost_diff"] = {}
    for a, b in combinations(names, 2):
        d = per[a]["cost"] - per[b]["cost"]
        lo, hi = bootstrap_ci(lambda i, d=d: d[i].mean(), n, args.block, args.n_boot, 2)
        verdict = "excludes 0" if lo > 0 or hi < 0 else "includes 0"
        out["paired_cost_diff"][f"{a} - {b}"] = {"mean": float(d.mean()), "ci95": [lo, hi]}
        print(f"   {LABELS[a]:<20} - {LABELS[b]:<20} {d.mean():+.4f}  CI [{lo:+.4f}, {hi:+.4f}]  ({verdict})")

    print(f"\n3. Extreme-hour recall vs k (battery full-power duration is about {args.battery_hours:.1f} h)")
    out["recall_by_k"] = {}
    print(f"   {'':<20}" + "".join(f"{'top-' + str(k):>9}" for k in K_VALUES) + "   |" + "".join(f"{'bot-' + str(k):>9}" for k in K_VALUES))
    for name in names:
        tops, bots = [], []
        for k in K_VALUES:
            pairs = [recall_k(f, a, k) for f, a in zip(per[name]["forecast"], per[name]["actual"])]
            tops.append(float(np.mean([p[0] for p in pairs]))); bots.append(float(np.mean([p[1] for p in pairs])))
        out["recall_by_k"][name] = {"top": dict(zip(K_VALUES, tops)), "bottom": dict(zip(K_VALUES, bots))}
        print(f"   {LABELS[name]:<20}" + "".join(f"{v:>9.2f}" for v in tops) + "   |" + "".join(f"{v:>9.2f}" for v in bots))

    print("\n4. Pooled daily correlations with -gap to oracle, day-block bootstrap")
    print("   (metrics are all oriented so higher = better forecast; neg_mae = -MAE)")
    gap = {c: per[c]["cost"] - shared["oracle"] for c in names}

    def pooled_corr(metric: str, idx: np.ndarray) -> float:
        x = np.concatenate([per[c][metric][idx] for c in names])
        y = np.concatenate([-gap[c][idx] for c in names])
        ok = ~np.isnan(x)
        return float(np.corrcoef(x[ok], y[ok])[0, 1])

    everyone = np.arange(n)
    out["pooled_corr_with_neg_gap"] = {}
    for metric in ("tau", "neg_mae", "top4"):
        r = pooled_corr(metric, everyone)
        lo, hi = bootstrap_ci(lambda i, m=metric: pooled_corr(m, i), n, args.block, args.n_boot, 3)
        out["pooled_corr_with_neg_gap"][metric] = {"r": r, "ci95": [lo, hi]}
        print(f"   corr({metric:<7}, -gap)  r={r:+.3f}  CI [{lo:+.3f}, {hi:+.3f}]")
    for m1, m2 in (("tau", "neg_mae"), ("top4", "tau"), ("top4", "neg_mae")):
        diff = pooled_corr(m1, everyone) - pooled_corr(m2, everyone)
        lo, hi = bootstrap_ci(lambda i, a=m1, b=m2: pooled_corr(a, i) - pooled_corr(b, i), n, args.block, args.n_boot, 4)
        out["pooled_corr_with_neg_gap"][f"{m1}_minus_{m2}"] = {"diff": diff, "ci95": [lo, hi]}
        print(f"   corr({m1}) - corr({m2}) = {diff:+.3f}  CI [{lo:+.3f}, {hi:+.3f}]")

    print("\n5. Same correlations after removing each day's common level (day-demeaned across candidates):")
    print("   isolates 'on the same day, does the candidate with better metric X cost less?' from")
    print("   'volatile days are hard for every model and leave a bigger gap to the oracle'.")

    def demeaned_corr(metric: str, idx: np.ndarray) -> float:
        x = np.stack([per[c][metric][idx] for c in names], 1)
        y = np.stack([-gap[c][idx] for c in names], 1)
        x = x - np.nanmean(x, 1, keepdims=True); y = y - y.mean(1, keepdims=True)
        ok = ~np.isnan(x)
        return float(np.corrcoef(x[ok], y[ok])[0, 1])

    out["day_demeaned_corr_with_neg_gap"] = {}
    for metric in ("tau", "neg_mae", "top4"):
        r = demeaned_corr(metric, everyone)
        lo, hi = bootstrap_ci(lambda i, m=metric: demeaned_corr(m, i), n, args.block, args.n_boot, 5)
        out["day_demeaned_corr_with_neg_gap"][metric] = {"r": r, "ci95": [lo, hi]}
        print(f"   corr({metric:<7}, -gap)  r={r:+.3f}  CI [{lo:+.3f}, {hi:+.3f}]")

    for m1, m2 in (("tau", "neg_mae"), ("top4", "neg_mae"), ("top4", "tau")):
        diff = demeaned_corr(m1, everyone) - demeaned_corr(m2, everyone)
        lo, hi = bootstrap_ci(lambda i, a=m1, b=m2: demeaned_corr(a, i) - demeaned_corr(b, i), n, args.block, args.n_boot, 6)
        out["day_demeaned_corr_with_neg_gap"][f"{m1}_minus_{m2}"] = {"diff": diff, "ci95": [lo, hi]}
        print(f"   corr({m1}) - corr({m2}) = {diff:+.3f}  CI [{lo:+.3f}, {hi:+.3f}]")

    print("\n6. Forecast sharpness and how much the battery actually trades")
    print("   The LP only cycles when predicted spread beats degradation cost plus round-trip losses, so a forecast")
    print("   that flattens the day trades little. (Within-day std of the price, EUR/kWh x1000; charged kWh per day.)")
    actual_std = float(per[names[0]]["actual"].std(axis=1).mean() * 1000)
    out["sharpness_and_cycling"] = {"actual_within_day_std_x1000": actual_std}
    print(f"   {'actual':<24} {actual_std:6.2f}")
    for name in names:
        fstd = float(per[name]["forecast"].std(axis=1).mean() * 1000)
        cyc = float(per[name]["charge"].sum(axis=1).mean())
        out["sharpness_and_cycling"][name] = {"forecast_within_day_std_x1000": fstd, "charged_kwh_per_day": cyc}
        print(f"   {LABELS[name]:<24} {fstd:6.2f}   charged {cyc:5.2f} kWh/day")

    if args.vs:
        _, other, _ = load_days(args.vs)
        print(f"\n7. Paired daily cost difference, this run minus {args.vs} (EUR/day; <0 = this run cheaper)")
        out["vs_other_run"] = {}
        for name in names:
            if name not in other:
                continue
            d = per[name]["cost"] - other[name]["cost"]
            lo, hi = bootstrap_ci(lambda i, d=d: d[i].mean(), n, args.block, args.n_boot, 7)
            out["vs_other_run"][name] = {"mean": float(d.mean()), "ci95": [lo, hi]}
            verdict = "excludes 0" if lo > 0 or hi < 0 else "includes 0"
            print(f"   {LABELS[name]:<24} {d.mean():+.4f}  CI [{lo:+.4f}, {hi:+.4f}]  ({verdict})")

    if args.out:
        Path(args.out).write_text(json.dumps(out, indent=2))
        print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
