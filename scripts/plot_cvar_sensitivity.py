"""Plot the CVaR sensitivity sweep results (scripts/cvar_sensitivity.py)."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt

from scripts.style import COLORS, GRID, INK, INK_MUTED, LABELS

RESULTS_PATH = Path("artifacts_real/cvar_sensitivity.json")
OUT_PATH = Path("docs/images/cvar_sensitivity.png")


def main() -> None:
    results = json.loads(RESULTS_PATH.read_text())
    weights = sorted(float(k.removeprefix("cvar_weight_")) for k in results if k.startswith("cvar_weight_"))
    pick = lambda key: [results[f"cvar_weight_{w}"][key] for w in weights]
    panels = [
        ("total_eur", f"Total realised cost, {results['days']} days (EUR)"),
        ("mean_worst_10pct_days_eur", "Average cost of the worst 10% of days (EUR/day)"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4))
    for ax, (key, label) in zip(axes, panels):
        values = pick(key)
        ax.plot(weights, values, marker="o", color=COLORS[results["candidate"]], linewidth=2)
        for w, v in zip(weights, values):
            ax.annotate(f"{v:.2f}", (w, v), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8.5, color=INK_MUTED)
        ax.set_xlabel("CVaR weight (0 = expected cost only, 1 = worst-case tail only)", color=INK_MUTED, fontsize=9)
        ax.set_ylabel(label, color=INK_MUTED)
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.set_ylim(0, max(values) * 1.25)  # zero-based, so a 0.01 change does not look like a collapse
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    fig.suptitle(f"CVaR risk aversion: {LABELS[results['candidate']]}, real-data backtest", fontsize=12, x=0.01, ha="left", color=INK)
    fig.tight_layout()
    fig.savefig(OUT_PATH, dpi=150)
    plt.close(fig)
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
