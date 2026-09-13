"""Plot the CVaR sensitivity sweep results (scripts/cvar_sensitivity.py)."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt

RESULTS_PATH = Path("artifacts_real/cvar_sensitivity.json")
OUT_PATH = Path("docs/images/cvar_sensitivity.png")


def main() -> None:
    results = json.loads(RESULTS_PATH.read_text())
    weights = sorted(float(k.removeprefix("cvar_weight_")) for k in results if k.startswith("cvar_weight_"))
    pct = [results[f"cvar_weight_{w}"]["vs_baseline_pct"] for w in weights]

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.axhline(0, color="#333333", linewidth=1, linestyle="-")
    ax.plot(weights, pct, marker="o", color="#2ca02c", linewidth=2)
    for w, p in zip(weights, pct):
        ax.annotate(f"{p:+.1f}%", (w, p), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=9)

    ax.set_xlabel("CVaR weight (0 = pure expected cost, 1 = pure worst-case)")
    ax.set_ylabel("Realised grid cost vs. no-battery baseline (%)")
    ax.set_title(f"{results['days']}-day real-data backtest: risk-aversion sensitivity")
    ax.text(
        0.02, 0.02,
        "0% = same as no battery; negative = battery cheaper; positive = battery more expensive",
        transform=ax.transAxes, fontsize=8, color="#555555",
    )

    fig.tight_layout()
    fig.savefig(OUT_PATH, dpi=150)
    plt.close(fig)
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
