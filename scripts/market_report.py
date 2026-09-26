"""Trading-desk style report for a market-side backtest (a merchant battery, no site load or PV).

Reads the per-day records of a completed run and reports, per forecaster:
  - revenue in EUR per MW per year and the share of the perfect-foresight (oracle) revenue captured,
  - the distribution of daily revenue (mean, spread, worst day, share of losing days, an
    annualised mean/std ratio) and the maximum drawdown of cumulative revenue,
  - energy cycled per day,
  - the same revenue and capture figures by calendar year (the regimes),
  - paired daily revenue differences between forecasters with block-bootstrap intervals,
  - how forecast metrics (MAE, Kendall's tau, top-k recall) relate to revenue, across forecasters
    and, day by day, within the same day.

With no site load or PV the no-battery baseline earns nothing, so market revenue is minus the
realised grid cost. The headline "revenue" is NET of the modelled degradation cost (the same
per-kWh throughput cost the optimiser pays), because that is what the schedule is optimised for;
gross market revenue is reported alongside. The oracle's own throughput is not stored in the run
records, so its schedule is re-solved from the stored actual prices (and its gross revenue is
checked against the stored value). "k" for the recall metrics defaults to 3 hours (a 2-hour
battery cycling up to 1.5 times a day trades about that many hours each way).

Several metrics files can be merged (candidates from later runs over the same days are added).

Usage:
    python -m scripts.market_report --metrics results/market/run_metrics_market_de_lu.json \\
        --out results/market/market_report_de_lu.json
    python -m scripts.market_report --metrics results/market/run_metrics_market_de_lu.json \\
        results/market/run_metrics_market_de_lu_aligned.json --out results/market/market_report_de_lu.json
"""
from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from scripts.robustness import bootstrap_ci, recall_k
from scripts.style import LABELS


def _config_for(metrics_path: str) -> Path:
    """configs/market_<market>.yaml for results/market/run_metrics_market_<market>[_suffix].json."""
    stem = Path(metrics_path).stem.removeprefix("run_metrics_")
    parts = stem.split("_")  # market, de, lu, ... or market, nl, ...
    for length in (3, 2):
        candidate = Path("configs") / ("_".join(parts[:length]) + ".yaml")
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"no configs/market_*.yaml matches {metrics_path}; pass --config")


def oracle_net_revenue(actual_prices: np.ndarray, config_path: Path, gross_check: np.ndarray, cache: Path) -> np.ndarray:
    """Net-of-degradation oracle revenue per day, re-solving the perfect-foresight LP on the stored
    actual prices (cached on disk). The gross revenue of these schedules must match the stored one."""
    if cache.exists():
        return np.array(json.loads(cache.read_text())["oracle_net_revenue_eur"])
    from battery_schedule.config import load_config
    from battery_schedule.optimise import solve_schedule

    cfg = load_config(config_path)["optimization"]
    net, gross = [], []
    for prices in actual_prices:
        scenario = np.stack([np.zeros_like(prices), prices], axis=-1)[None]
        sched = solve_schedule(scenario, np.zeros_like(prices), cfg.export_price_ratio * prices, cfg, cvar_weight=0.0)
        charge, discharge = sched.charge_kw.to_numpy(), sched.discharge_kw.to_numpy()
        revenue = float((discharge * cfg.export_price_ratio * prices - charge * prices).sum())
        gross.append(revenue)
        net.append(revenue - cfg.degradation_eur_per_kwh * float((charge + discharge).sum()))
    if not np.allclose(gross, gross_check, rtol=1e-4, atol=1e-6):
        raise RuntimeError("re-solved oracle does not reproduce the stored oracle revenue")
    cache.write_text(json.dumps({"oracle_net_revenue_eur": net}))
    return np.array(net)


def load(metrics_paths: list[str] | str) -> tuple[dict, list[str], dict[str, dict[str, np.ndarray]], np.ndarray]:
    paths = [metrics_paths] if isinstance(metrics_paths, str) else list(metrics_paths)
    backtest, daily = None, []
    seen = set()
    for path in paths:
        run = json.loads(Path(path).read_text())["backtest"]
        backtest = backtest or run
        daily += [d for d in run["daily"] if (d["date"], d["candidate"]) not in seen]
        seen |= {(d["date"], d["candidate"]) for d in run["daily"]}
    candidates = list(dict.fromkeys(d["candidate"] for d in daily))
    dates = sorted(set.intersection(*({d["date"] for d in daily if d["candidate"] == c} for c in candidates)))
    deg = backtest["battery"]["degradation_eur_per_kwh"]
    by_key = {(d["date"], d["candidate"]): d for d in daily}
    per: dict[str, dict[str, np.ndarray]] = {}
    for name in candidates:
        ordered = [by_key[(t, name)] for t in dates]
        gross = np.array([r["baseline_cost_eur"] - r["realised_cost_eur"] for r in ordered])
        throughput = np.array([sum(r["charge_kw"]) + sum(r["discharge_kw"]) for r in ordered])
        per[name] = {
            "gross_revenue": gross, "revenue": gross - deg * throughput,
            "tau": np.array([np.nan if r["kendall_tau"] is None else r["kendall_tau"] for r in ordered]),
            "neg_mae": -np.array([r["mae_price_eur_kwh"] for r in ordered]),
            "forecast": np.array([r["forecast_price_eur_kwh"] for r in ordered]),
            "actual": np.array([r["actual_price_eur_kwh"] for r in ordered]),
            "discharge": np.array([r["discharge_kw"] for r in ordered]),
        }
    first = {d["date"]: d for d in daily}
    gross_oracle = np.array([first[t]["baseline_cost_eur"] - first[t]["oracle_cost_eur"] for t in dates])
    config_path = _config_for(paths[0])
    cache = Path(paths[0]).with_name(f"oracle_net_{Path(config_path).stem.removeprefix('market_')}.json")
    oracle = oracle_net_revenue(per[candidates[0]]["actual"], config_path, gross_oracle, cache) if len(dates) else gross_oracle
    return backtest, dates, per, oracle


def max_drawdown(daily_revenue: np.ndarray) -> float:
    cumulative = np.cumsum(daily_revenue)
    return float((np.maximum.accumulate(cumulative) - cumulative).max())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", nargs="+", required=True)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--block", type=int, default=14, help="bootstrap block length in evaluated days")
    parser.add_argument("--n-boot", type=int, default=3000)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    backtest, dates, per, oracle = load(args.metrics)
    names = list(per)
    power_mw = backtest["battery"]["power_kw"] / 1000
    capacity_kwh = backtest["battery"]["capacity_kwh"]
    stride = backtest["stride_days"]
    n = len(dates)
    years = np.array([d[:4] for d in dates])
    scale = 365 / power_mw  # mean EUR per evaluated day -> EUR per MW per year
    out: dict = {"days_evaluated": n, "first_day": dates[0], "last_day": dates[-1], "stride_days": stride,
                 "power_mw": power_mw, "capacity_mwh": capacity_kwh / 1000, "k": args.k}
    print(f"{n} evaluated days, {dates[0]} to {dates[-1]} (every {stride} day(s)); "
          f"{power_mw:g} MW / {capacity_kwh / 1000:g} MWh; block bootstrap {args.block} days\n")

    oracle_year = float(oracle.mean() * scale)
    print(f"Perfect-foresight oracle, net of degradation: {oracle_year:,.0f} EUR/MW/year\n")
    out["oracle_eur_per_mw_year"] = oracle_year

    print("1. Revenue and risk (EUR per MW; net of the modelled degradation cost unless stated)")
    print(f"   {'':<40}{'net EUR/MW/yr':>14}{'gross':>9}{'capture':>9}{'daily std':>10}{'p5 day':>9}{'worst day':>10}{'loss days':>10}{'max DD':>9}{'cycles/day':>11}")
    out["summary"] = {}
    for name in names:
        rev = per[name]["revenue"] / power_mw
        cycles = float((per[name]["discharge"].sum(axis=1) / capacity_kwh).mean())
        row = {
            "net_eur_per_mw_year": float(rev.mean() * 365), "gross_eur_per_mw_year": float(per[name]["gross_revenue"].mean() * scale),
            "capture_ratio": float(per[name]["revenue"].sum() / oracle.sum()),
            "daily_std_eur_per_mw": float(rev.std()), "p5_day_eur_per_mw": float(np.percentile(rev, 5)),
            "worst_day_eur_per_mw": float(rev.min()), "share_losing_days": float((rev < -1e-9).mean()),
            "max_drawdown_eur_per_mw": max_drawdown(rev), "cycles_per_day": cycles,
        }
        out["summary"][name] = row
        print(f"   {LABELS[name]:<40}{row['net_eur_per_mw_year']:>14,.0f}{row['gross_eur_per_mw_year']:>9,.0f}{row['capture_ratio']:>9.1%}"
              f"{row['daily_std_eur_per_mw']:>10.1f}{row['p5_day_eur_per_mw']:>9.1f}{row['worst_day_eur_per_mw']:>10.1f}"
              f"{row['share_losing_days']:>10.1%}{row['max_drawdown_eur_per_mw']:>9,.0f}{cycles:>11.2f}")

    print("\n2. By calendar year (EUR/MW/year; capture in brackets)")
    out["by_year"] = {}
    unique_years = sorted(set(years))
    print(f"   {'':<40}" + "".join(f"{y:>16}" for y in unique_years))
    oracle_by_year = {y: float(oracle[years == y].mean() * scale) for y in unique_years}
    print(f"   {'oracle':<40}" + "".join(f"{oracle_by_year[y]:>16,.0f}" for y in unique_years))
    out["by_year"]["oracle"] = oracle_by_year
    for name in names:
        cells, block = [], {}
        for y in unique_years:
            mask = years == y
            value = float(per[name]["revenue"][mask].mean() * scale)
            capture = float(per[name]["revenue"][mask].sum() / oracle[mask].sum())
            block[y] = {"eur_per_mw_year": value, "capture_ratio": capture}
            cells.append(f"{value:>9,.0f} ({capture:>4.0%})")
        out["by_year"][name] = block
        print(f"   {LABELS[name]:<40}" + "".join(f"{c:>16}" for c in cells))

    print("\n3. Paired daily revenue differences a - b (EUR/MW/year, 95% block-bootstrap CI; >0 = a earns more)")
    out["paired_revenue_diff"] = {}
    for a, b in combinations(names, 2):
        d = (per[a]["revenue"] - per[b]["revenue"]) / power_mw
        lo, hi = bootstrap_ci(lambda i, d=d: d[i].mean() * 365, n, args.block, args.n_boot, 11)
        out["paired_revenue_diff"][f"{a} - {b}"] = {"eur_per_mw_year": float(d.mean() * 365), "ci95": [lo, hi]}
        tag = "excludes 0" if lo > 0 or hi < 0 else "includes 0"
        print(f"   {LABELS[a]:<40} - {LABELS[b]:<40}{d.mean() * 365:>+9,.0f}  [{lo:>+8,.0f}, {hi:>+8,.0f}]  ({tag})")

    print(f"\n4. Forecast quality (k = {args.k})")
    metrics: dict[str, dict[str, np.ndarray]] = {}
    for name in names:
        recalls = np.array([recall_k(f, a, args.k) for f, a in zip(per[name]["forecast"], per[name]["actual"])])
        metrics[name] = {"tau": per[name]["tau"], "neg_mae": per[name]["neg_mae"], "top_k": recalls[:, 0], "bottom_k": recalls[:, 1]}
    out["forecast_quality"] = {}
    print(f"   {'':<40}{'MAE EUR/MWh':>12}{'tau':>8}{f'top-{args.k}':>8}{f'bot-{args.k}':>8}")
    for name in names:
        row = {"mae_eur_per_mwh": float(-metrics[name]["neg_mae"].mean() * 1000), "tau": float(np.nanmean(metrics[name]["tau"])),
               "top_k_recall": float(metrics[name]["top_k"].mean()), "bottom_k_recall": float(metrics[name]["bottom_k"].mean())}
        out["forecast_quality"][name] = row
        print(f"   {LABELS[name]:<40}{row['mae_eur_per_mwh']:>12.2f}{row['tau']:>8.3f}{row['top_k_recall']:>8.2f}{row['bottom_k_recall']:>8.2f}")

    print("\n5. Across forecasters: Spearman correlation of each metric with total revenue (n = number of forecasters)")
    totals = [per[c]["revenue"].sum() for c in names]
    out["model_level_spearman_with_revenue"] = {}
    for key, values in (("neg_mae", [metrics[c]["neg_mae"].mean() for c in names]),
                        ("tau", [np.nanmean(metrics[c]["tau"]) for c in names]),
                        ("top_k", [metrics[c]["top_k"].mean() for c in names])):
        rho = float(spearmanr(values, totals).statistic)
        out["model_level_spearman_with_revenue"][key] = rho
        print(f"   {key:<8} rho = {rho:+.2f}")

    print("\n6. Same day, across forecasters: day-demeaned correlation of each metric with revenue (block bootstrap over days)")

    def demeaned(metric: str, idx: np.ndarray) -> float:
        x = np.stack([metrics[c][metric][idx] for c in names], 1)
        y = np.stack([per[c]["revenue"][idx] for c in names], 1)
        x = x - np.nanmean(x, 1, keepdims=True); y = y - y.mean(1, keepdims=True)
        ok = ~np.isnan(x)
        return float(np.corrcoef(x[ok], y[ok])[0, 1])

    everyone = np.arange(n)
    out["day_demeaned_corr_with_revenue"] = {}
    for metric in ("tau", "neg_mae", "top_k"):
        r = demeaned(metric, everyone)
        lo, hi = bootstrap_ci(lambda i, m=metric: demeaned(m, i), n, args.block, args.n_boot, 12)
        out["day_demeaned_corr_with_revenue"][metric] = {"r": r, "ci95": [lo, hi]}
        print(f"   corr({metric:<7}, revenue)  r = {r:+.3f}  CI [{lo:+.3f}, {hi:+.3f}]")
    for m1, m2 in (("tau", "neg_mae"), ("top_k", "neg_mae")):
        diff = demeaned(m1, everyone) - demeaned(m2, everyone)
        lo, hi = bootstrap_ci(lambda i, a=m1, b=m2: demeaned(a, i) - demeaned(b, i), n, args.block, args.n_boot, 13)
        out["day_demeaned_corr_with_revenue"][f"{m1}_minus_{m2}"] = {"diff": diff, "ci95": [lo, hi]}
        print(f"   corr({m1}) - corr({m2}) = {diff:+.3f}  CI [{lo:+.3f}, {hi:+.3f}]")

    print("\n7. Where the forecast puts the day's peak (share of days the highest predicted price falls in each window)")
    windows = {"midday 11-14h": (11, 14), "evening 16-20h": (16, 20), "morning 6-9h": (6, 9)}
    actual = per[names[0]]["actual"]
    out["peak_hour_share"] = {"actual": {}}
    line = f"   {'':<40}" + "".join(f"{w:>16}" for w in windows)
    print(line)
    for label, series in (("actual", actual), *((c, per[c]["forecast"]) for c in names)):
        argmax = series.argmax(axis=1)
        shares = {w: float(((argmax >= lo) & (argmax <= hi)).mean()) for w, (lo, hi) in windows.items()}
        out["peak_hour_share"][label] = shares
        print(f"   {LABELS.get(label, label):<40}" + "".join(f"{shares[w]:>16.0%}" for w in windows))

    if args.out:
        Path(args.out).write_text(json.dumps(out, indent=2))
        print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
