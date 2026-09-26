#!/bin/bash
# Finish the market-side study: the second-pass backtests, then the reports and charts.
# Everything here reads local files only (data/raw/market_*.csv); no network is needed.
#
# Before running:
#   - the FR backtest must be finished (this script refuses to start while a backtest is running);
#   - keep the laptop lid open and plugged in (a sleeping machine pauses the runs).
#
# Usage, from anywhere:
#   bash scripts/run_market_study.sh            # everything (about 2 h 15 min)
#   bash scripts/run_market_study.sh reports    # only the reports and charts (a few minutes)
#
# Logs go to logs/. If a step fails the script stops; rerun it and finished steps are skipped.
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
mkdir -p logs results/market

if pgrep -f "battery_schedule.cli run" > /dev/null; then
  echo "A backtest is still running (probably FR). Wait for it to finish, then run this again."; exit 1
fi

run_backtest() {  # $1 = config name, $2 = artifacts dir, $3 = destination file in results/market
  if [ -f "results/market/$3" ]; then echo "skip $1 (results/market/$3 already exists)"; return; fi
  echo "[$(date +%H:%M)] running $1 ..."
  caffeinate -i -s python -u -m battery_schedule.cli run --config "configs/$1.yaml" > "logs/$1.log" 2>&1
  cp "$2/run_metrics.json" "results/market/$3"
  echo "[$(date +%H:%M)] done $1"
}

if [ "${1:-all}" != "reports" ]; then
  run_backtest market_de_lu_aligned artifacts_market_de_lu_aligned run_metrics_market_de_lu_aligned.json
  run_backtest market_nl_aligned artifacts_market_nl_aligned run_metrics_market_nl_aligned.json
fi

# Keep the finished FR run and, if present, make sure DE-LU and NL main runs are in results/market.
for m in de_lu nl fr; do
  if [ ! -f "results/market/run_metrics_market_$m.json" ] && [ -f "artifacts_market_$m/run_metrics.json" ]; then
    cp "artifacts_market_$m/run_metrics.json" "results/market/run_metrics_market_$m.json"
  fi
done

# Consistency check: seasonal naive appears in both passes and must be identical.
python - <<'EOF'
import json
from pathlib import Path
for m in ("de_lu", "nl"):
    a, b = Path(f"results/market/run_metrics_market_{m}.json"), Path(f"results/market/run_metrics_market_{m}_aligned.json")
    if not (a.exists() and b.exists()):
        print(f"{m}: skipping naive check (a pass is missing)"); continue
    rows = lambda p: {d["date"]: d["realised_cost_eur"] for d in json.loads(p.read_text())["backtest"]["daily"] if d["candidate"] == "seasonal_naive"}
    x, y = rows(a), rows(b)
    diff = max(abs(x[t] - y[t]) for t in x.keys() & y.keys())
    print(f"{m}: seasonal naive, max |cost difference| between the two passes = {diff:.2e} EUR over {len(x.keys() & y.keys())} days")
EOF

merged() {  # main file plus the aligned pass when it exists
  local m=$1; local f="results/market/run_metrics_market_$m.json"
  [ -f "results/market/run_metrics_market_${m}_aligned.json" ] && f="$f results/market/run_metrics_market_${m}_aligned.json"
  echo "$f"
}

for m in de_lu nl fr; do
  [ -f "results/market/run_metrics_market_$m.json" ] || { echo "no results for $m yet, skipping its report"; continue; }
  python -m scripts.market_report --metrics $(merged $m) --out "results/market/market_report_$m.json" | tee "results/market/report_$m.txt" > /dev/null
  echo "wrote results/market/report_$m.txt"
done

ARGS=""
for m in de_lu nl fr; do
  [ -f "results/market/run_metrics_market_$m.json" ] || continue
  ARGS="$ARGS $m=$(merged $m | tr ' ' ',')"
done
python -m scripts.plot_market --metrics $ARGS
title() { case $1 in de_lu) echo "Germany-Luxembourg";; nl) echo "Netherlands";; fr) echo "France";; esac; }
PANEL_ARGS=()
for m in de_lu nl fr; do
  [ -f "results/market/run_metrics_market_$m.json" ] && PANEL_ARGS+=(--panel "$(title $m)=$(merged $m | tr ' ' ',')")
done
python -m scripts.plot_hourly_profile --out market_hourly_profile.png --unit EUR/MWh --scale 1000 "${PANEL_ARGS[@]}"
echo "All done. Send me results/market/report_*.txt (or just tell me it finished) and I will write up the README."
