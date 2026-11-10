"""
Summarise evaluation results across commuters.

Each commuter has an independent history, so evaluate_v4.py is run once per commuter folder:
  python3 evaluate_v4.py <commuter_1_dir> --out-dir out/v1
  python3 evaluate_v4.py <commuter_2_dir> --out-dir out/v2
Then:
  python3 summarize_commuters.py out/v1/evaluation_v4.json out/v2/evaluation_v4.json [--history none]
Prints the mean, lowest and highest value of each measure across commuters, and each commuter's own value.
"""
import argparse, json, os
import numpy as np
ap = argparse.ArgumentParser(); ap.add_argument("files", nargs="+"); ap.add_argument("--history", default="none", help="which history length to summarise (none, 14d, 7d, ...)")
A = ap.parse_args()
KEYS = [("trips", "Trips judged"), ("ap_model_2", "Forecast AP (Major+)"), ("ap_persist_2", "Baseline AP (Major+)"), ("ece_2", "Calibration error (Major+)"),
        ("conf_coverage", "Conformal coverage"), ("false_alarms_per_trip_forecast", "False alarms/trip, forecast"), ("false_alarms_per_trip_persist", "False alarms/trip, baseline")]
rows = []
for f in A.files:
    r = json.load(open(f))["expiry_sweep"][A.history]; rows.append((os.path.dirname(f) or f, r))
print(f"{len(rows)} commuter(s), history length: {A.history}")
print(f"{'measure':34s}{'mean':>9s}{'lowest':>9s}{'highest':>9s}   per commuter")
for k, name in KEYS:
    values = [r.get(k) for _, r in rows]
    available = [v for v in values if v is not None and np.isfinite(v)]
    stats = [np.mean(available), min(available), max(available)] if available else [None] * 3
    cell = lambda value: f"{value:.3f}" if value is not None and np.isfinite(value) else "-"
    print(f"{name:34s}" + "".join(f"{cell(v):>9s}" for v in stats) + "   " + ", ".join(map(cell, values)))
