"""
Export one real held-out trip with every model output, one row per GPS fix.
Reuses the model definitions in pipeline2.py (everything above the 'rolling-origin folds' section).
Run: python3 export_sample.py <dataset_dir> <out_csv>
"""
import sys
src = open(__file__.replace("export_sample.py", "pipeline2.py")).read()
head = src[:src.index("# ----------------------------------------------------------------------------- 1) rolling-origin folds")]
OUT_PATH = sys.argv[2]
sys.argv = [sys.argv[0], sys.argv[1], "/dev/null"]
exec(compile(head, "pipeline2_head", "exec"))

import xgboost as xgb
k = 48
trn, van, ten = order[:k - 6], order[k - 6:k], order[k:k + 6]
res, D, c, mdl = run_fold(trn, van, ten, injection=False, want=True)
m, cols = mdl["xgb"]
te = D[D.trip.isin(ten)].copy()
te["risk"] = m.predict_proba(te[cols])[:, 1]
te["safety_index"] = (100 * (1 - te.risk)).round(1)
contrib = m.get_booster().predict(xgb.DMatrix(te[cols]), pred_contribs=True)[:, :-1]
nice = {"d": "distance from route line", "d_hist": "distance from earlier trips", "excess_dwell": "stop time beyond usual here",
        "dwell": "seconds stopped", "v": "speed", "v_mean6": "average speed (last 60 s)", "v_resid": "speed unusual for this spot",
        "s": "position on route", "elapsed": "time since trip start", "hour": "time of day", "dir": "direction",
        "turn_sum6": "recent turning", "a_absmax6": "recent sharp speed change", "stops6": "recent stops", "band_poly": "deviation band (route)",
        "band_hist": "deviation band (history)", "heading_dev": "heading off the road direction", "dev_run": "time off-route",
        "stop_unexp": "stopped where trips rarely stop", "cum_stop": "total time stopped so far", "v_std6": "speed variability",
        "a": "acceleration", "jerk": "jerk", "turn": "turn rate", "a_min6": "recent braking", "dow": "day of week"}
def reasons(row):
    idx = np.argsort(-np.abs(row))[:2]
    return "; ".join(f"{nice.get(cols[i], cols[i])} {'raises' if row[i] > 0 else 'lowers'} risk" for i in idx)
te["top_reasons"] = [reasons(r) for r in contrib]
te["unusual_now"] = te.ev_if
te["unusualness_score"] = te.if2.round(3)
te["deviation_band"] = np.maximum(te.band_hist, te.band_poly).map({0: "Normal", 1: "Minor", 2: "Major", 3: "Critical"})
te["unusual_in_next_60s_actual"] = fwd(D, "ev_if")[te.index]
te["speed_kmh"] = (te.v * 3.6).round(1)
te["time_local"] = te.t.dt.tz_convert("Asia/Manila").dt.strftime("%Y-%m-%d %H:%M:%S")
te["latitude"] = LAT0 + te.y / KY; te["longitude"] = LON0 + te.x / KX
te["latitude"] = te.latitude.round(6); te["longitude"] = te.longitude.round(6)
te["risk"] = te.risk.round(3)
out_cols = ["time_local", "latitude", "longitude", "speed_kmh", "risk", "safety_index", "unusualness_score", "unusual_now",
            "deviation_band", "top_reasons", "unusual_in_next_60s_actual"]
# pick the held-out trip with the most unusual moments
pick = te.groupby("trip").unusual_in_next_60s_actual.sum().idxmax()
sample = te[te.trip == pick][out_cols]
sample.to_csv(OUT_PATH, index=False)
print("trip", pick, "direction", int(te[te.trip == pick].dir.iloc[0]), "rows", len(sample))
print("threshold tau", round(res["tau"], 3))
