"""Replay one commuter’s GPS trips and evaluate model-defined and reviewed labels."""
import argparse, json, os, sys, time, html, platform
from importlib.metadata import version
import numpy as np, pandas as pd
from sklearn.metrics import average_precision_score, confusion_matrix

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import model_v4 as M

ap = argparse.ArgumentParser()
ap.add_argument("dataset"); ap.add_argument("--labels"); ap.add_argument("--out-dir", default=os.path.join(HERE, "..", "results"))
ap.add_argument("--expiry", default="none,14,7", help="comma list of history lengths in days; 'none' = keep everything")
ap.add_argument("--minh", type=int, default=5, help="fewest earlier trips needed before a trip is judged")
ap.add_argument("--alert-level", type=int, choices=(1, 2, 3), default=2); ap.add_argument("--alert-prob", type=float, default=0.5)
ap.add_argument("--offroute-level", action="store_true", help="add a concern level from the length of off-route episodes, judged against the commuter's own earlier episodes")
ap.add_argument("--horizon", type=float, default=60.0, help="forecast window in seconds (sensitivity check)")
ap.add_argument("--tod-mode", default="four", choices=["four", "two", "none"], help="parts of the day used by the speed baseline (sensitivity check)")
ap.add_argument("--route-mode", default="history", choices=["history", "max"], help="route distance rule: planned route plus supported variants (history), or the stricter distance after warm-up (max)")
ap.add_argument("--route-min", type=int, default=5, help="earlier trips in the same direction before distance from earlier trips is used")
ap.add_argument("--model-opts", default="{}", help="JSON with forecaster settings to try: xgb (dict), drop (list), platt (bool), cal_share (float), pos_weight (bool), kind (xgb or logistic)")
ap.add_argument("--seg", type=float, default=100.0, help="road stretch length in metres (sensitivity check)")
ap.add_argument("--shrink", type=float, default=8.0, help="shrinkage K for time-of-day speed baselines (sensitivity check)")
ap.add_argument("--max-gap", type=float, default=25.0, help="seconds without a fix that break GPS coverage")
ap.add_argument("--event-window", type=float, default=120.0, help="maximum event matching tolerance in seconds")
ap.add_argument("--merge-s", type=float, default=60.0, help="alerts closer than this belong to one episode")
ARGS = ap.parse_args()
if ARGS.minh < 4:
    ap.error("--minh must be at least 4 to leave at least two fitting and two calibration trips")
if min(ARGS.horizon, ARGS.max_gap, ARGS.event_window) <= 0 or ARGS.merge_s < 0:
    ap.error("horizon, maximum gap and event window must be positive, and merge time nonnegative")
if not 0 < ARGS.alert_prob <= 1:
    ap.error("--alert-prob must be above 0 and at most 1")
M.MAX_GAP_S = ARGS.max_gap

_o = json.loads(ARGS.model_opts)
if "xgb" in _o: M.XGB_KW = dict(M.XGB_KW, **_o["xgb"])
if "drop" in _o: M.DROP_FEATURES = _o["drop"]
if "platt" in _o: M.USE_PLATT = _o["platt"]
if "cal_share" in _o: M.CAL_SHARE = _o["cal_share"]
if "pos_weight" in _o: M.POS_WEIGHT = _o["pos_weight"]
if "kind" in _o: M.MODEL_KIND = _o["kind"]
M.ROUTE_MODE = ARGS.route_mode; M.ROUTE_MIN_TRIPS = ARGS.route_min; M.TOD_MODE = ARGS.tod_mode; M.HORIZON_S = ARGS.horizon; M.SEG = ARGS.seg; M.SHRINK_K = ARGS.shrink; M.OFFROUTE_LEVEL = ARGS.offroute_level
D = M.load_dataset(ARGS.dataset)
EXPIRIES = [None if e.strip().lower() == "none" else float(e) for e in ARGS.expiry.split(",")]

# ------------------------------------------------------------------ replay
TIMES = dict(fit=[], score=[])
def replay(expiry_days):
    rows = []
    for pos, i in enumerate(D.order):
        hist = [j for j in D.order[:pos] if D.end[j] < D.dep[i] and
                (expiry_days is None or (D.dep[i] - D.dep[j]).total_seconds() <= expiry_days * 86400)]
        if len(hist) < ARGS.minh: continue
        _t0 = time.perf_counter(); _m = M.fit_model(D, hist); _t1 = time.perf_counter(); out = M.predict_trip(D, _m, i); _t2 = time.perf_counter()
        TIMES['fit'].append(_t1 - _t0); TIMES['score'].append(_t2 - _t1)
        out["file"] = D.files[i]; rows.append(out)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(
        columns=["trip", "file", "t", "observed", "segment", "dt", "lvl_now", "lvl_fwd", "P1", "P2", "P3", "d"])

# ------------------------------------------------------------------ metrics
def ece(y, p, bins=8):
    if len(y) == 0: return np.nan
    b = np.minimum((p * bins).astype(int), bins - 1); e = 0.0
    for k in range(bins):
        s = b == k
        if s.any(): e += s.mean() * abs(y[s].mean() - p[s].mean())
    return float(e)

def ap_score(y, s): return float(average_precision_score(y, s)) if 0 < y.sum() < len(y) else np.nan

def episode_starts(g, flag):
    times = g.loc[flag, "t"]
    segments = g.loc[flag, "segment"]
    starts = times.diff().dt.total_seconds().gt(ARGS.merge_s) | segments.ne(segments.shift())
    return times[starts].tolist()

def alert_forecast(df):
    return df[f"P{ARGS.alert_level}"] >= ARGS.alert_prob

def alert_now(df):
    return df.lvl_now >= ARGS.alert_level

def alert_fixed(df):
    return df.observed & (df.d > M.BANDS[0])

def binary_metrics(y, predicted):
    tn, fp, fn, tp = map(int, confusion_matrix(y, predicted, labels=[False, True]).ravel()) if len(y) else (0, 0, 0, 0)
    return dict(matrix=[[tn, fp], [fn, tp]], precision=tp / (tp + fp) if tp + fp else None,
                recall=tp / (tp + fn) if tp + fn else None,
                specificity=tn / (tn + fp) if tn + fp else None)

def summarize(df, verified_normal=()):
    if df.empty:
        return {}
    r = dict(trips=int(df.trip.nunique()), rows=len(df))
    r["mean_history_trips"] = float(df.groupby("trip").n_hist.first().mean())
    eligible = df.dropna(subset=["lvl_fwd", "pred_level"])
    r["scored_windows"] = len(eligible)
    r["unscored_windows"] = len(df) - len(eligible)
    r["alert_matrices"] = {}
    for k in (1, 2, 3):
        y = eligible.lvl_fwd >= k
        p = eligible[f"P{k}"]
        persistence = eligible.lvl_now >= k
        new = ~persistence
        r[f"base_rate_{k}"] = float(y.mean()) if len(y) else None
        r[f"ap_model_{k}"] = ap_score(y, p)
        r[f"ap_persist_{k}"] = ap_score(y, persistence.astype(float))
        r[f"ece_{k}"] = ece(y.to_numpy(), p.to_numpy())
        r["alert_matrices"][str(k)] = dict(
            forecast=binary_metrics(y, p >= ARGS.alert_prob),
            persistence=binary_metrics(y, persistence),
            fixed_distance=binary_metrics(y, alert_fixed(eligible)),
            forecast_from_below_threshold=binary_metrics(y[new], p[new] >= ARGS.alert_prob),
        )
    r["ordinal_matrix"] = (confusion_matrix(eligible.lvl_fwd, eligible.pred_level, labels=[0, 1, 2, 3]).tolist()
                           if len(eligible) else np.zeros((4, 4), dtype=int).tolist())
    r["mae_model"] = float((eligible.exp_level - eligible.lvl_fwd).abs().mean())
    r["mae_persist"] = float((eligible.lvl_now - eligible.lvl_fwd).abs().mean())
    sets = eligible[eligible.conf_set.notna()]
    r["conf_set_windows"] = len(sets)
    r["conf_coverage"] = float(np.mean([y in s for y, s in zip(sets.lvl_fwd, sets.conf_set)])) if len(sets) else None
    r["conf_avg_size"] = float(np.mean(sets.conf_set.map(len))) if len(sets) else None
    ordinary = df[df.file.isin(verified_normal) & df.observed]
    hours = ordinary.dt.sum() / 3600
    r["ordinary_trips"] = ordinary.trip.nunique()
    for name, alert in (("forecast", alert_forecast), ("persist", alert_now), ("fixed", alert_fixed)):
        counts = [len(episode_starts(g, alert(g))) for _, g in ordinary.groupby("trip")]
        r[f"false_alarms_per_trip_{name}"] = float(np.mean(counts)) if counts else None
        r[f"alerts_per_hour_{name}"] = sum(counts) / hours if hours else None
        r[f"trips_with_alarm_{name}"] = float(np.mean(np.array(counts) > 0)) if counts else None
    return r

# ------------------------------------------------------------------ independently reviewed unusual movement
def load_notes(path):
    if not path:
        return None
    L = pd.read_csv(path)
    if not {"file", "incident"}.issubset(L.columns):
        ap.error("--labels requires file and incident columns")
    incident = pd.to_numeric(L.incident, errors="coerce")
    if (L.incident.notna() & ~incident.isin([0, 1])).any() or L.file.isna().any():
        ap.error("--labels requires a file on each row and incident values of 0, 1 or blank")
    L["incident"] = incident
    for column in ("incident_time", "end_time"):
        L[column] = pd.to_datetime(L.get(column, pd.Series(index=L.index, dtype=str)),
                                   format="mixed", utc=True, errors="coerce")
    L["event_type"] = L.get("event_type", pd.Series("", index=L.index)).fillna("").str.strip().str.lower()
    positive = L.incident.eq(1)
    negative = L.incident.eq(0)
    if set(L.file[positive]) & set(L.file[negative]):
        ap.error("--labels cannot mark the same trip as both ordinary and event-positive")
    if (L.end_time < L.incident_time).any():
        ap.error("--labels event end cannot precede onset")
    return L

def field_check(df, notes, alert):
    event_types = ["unexpected_stop", "route_deviation"]
    verified = notes.incident.eq(1) & notes.event_type.isin(event_types) & notes.incident_time.notna()
    events = notes[verified].drop_duplicates(["file", "incident_time", "event_type"])
    unverified = set(notes.loc[notes.incident.eq(1) & ~verified, "file"])
    normals = set(notes.loc[notes.incident.eq(0), "file"])
    y, predicted, matches, leads = [], [], [], []
    unmatched, unscored_events, hours = 0, 0, 0.0
    for file, g in df.groupby("file"):
        if file in normals and g.observed.any():
            starts = episode_starts(g, alert(g))
            y.append(False)
            predicted.append(bool(starts))
            unmatched += len(starts)
            hours += g.loc[g.observed, "dt"].sum() / 3600
            continue
        trip_events = events[events.file.eq(file)].sort_values("incident_time")
        if trip_events.empty:
            continue
        starts = episode_starts(g, alert(g))
        observed = g.loc[g.observed, "t"]
        eligible = []
        for event in trip_events.itertuples():
            onset = event.incident_time
            begin = onset - pd.Timedelta(seconds=min(ARGS.horizon, ARGS.event_window))
            end = event.end_time if pd.notna(event.end_time) else onset + pd.Timedelta(seconds=ARGS.event_window)
            if not observed.between(begin, onset, inclusive="left").any() or onset > g.t.iloc[-1]:
                unscored_events += 1
                matches.append(dict(file=file, onset=str(onset), status="no observed pre-onset window"))
            else:
                eligible.append((begin, onset, end))
        early = {}
        # Reserve early matches before assigning any late detections.
        for i, (begin, onset, _) in enumerate(eligible):
            matched = next((t for t in starts if begin <= t < onset), None)
            if matched is not None:
                early[i] = matched
                starts.remove(matched)
                leads.append((onset - matched).total_seconds())
        for i, (_, onset, end) in enumerate(eligible):
            matched = early.get(i)
            status = "early"
            if matched is None:
                matched = next((t for t in starts if onset <= t <= end), None)
                status = "late" if matched is not None else "missed"
                if matched is not None:
                    starts.remove(matched)
            matches.append(dict(file=file, onset=str(onset), status=status,
                                alert_time=str(matched) if matched is not None else None))
        if eligible:
            y.append(True)
            predicted.append(bool(early))
            if file not in unverified and len(eligible) == len(trip_events):
                unmatched += len(starts)
                hours += g.loc[g.observed, "dt"].sum() / 3600
    unscored_events += int((~events.file.isin(df.file)).sum()) if not df.empty else len(events)
    return dict(trip_matrix=binary_metrics(y, predicted), scored_trips=len(y), confirmed_events=len(events),
                unscored_events=unscored_events, excluded_positive_notes=int((notes.incident.eq(1) & ~verified).sum()),
                early_events=len(leads), late_events=sum(m["status"] == "late" for m in matches),
                missed_events=sum(m["status"] == "missed" for m in matches),
                median_lead_s=float(np.median(leads)) if leads else None,
                unmatched_alerts=unmatched, reviewed_hours=hours,
                unmatched_alerts_per_hour=unmatched / hours if hours else None, details=matches)

# ------------------------------------------------------------------ run
t0 = time.time()
NOTES = load_notes(ARGS.labels)
normal_files = set(NOTES.loc[NOTES.incident.eq(0), "file"]) if NOTES is not None else set()
RES = {}
for e in dict.fromkeys([None, *EXPIRIES]):
    name = "none" if e is None else f"{e:g}d"
    print(f"replaying with history length: {name} ...", flush=True)
    RES[name] = replay(e)

# Freeze the no-expiry pseudo-labels across history comparisons. Independent field labels
# remain the primary reference for measuring alerts.
base = RES["none"]
common = set(base.trip) if not base.empty else set()
names = ["none" if e is None else f"{e:g}d" for e in EXPIRIES]
for name in names:
    frame = RES[name]
    common &= set(frame.trip) if not frame.empty else set()
    if name != "none" and not frame.empty:
        reference = base.set_index(["trip", "t"]).lvl_fwd
        frame["lvl_fwd_own_history"] = frame.lvl_fwd
        frame["lvl_fwd"] = reference.reindex(pd.MultiIndex.from_frame(frame[["trip", "t"]])).to_numpy()

manifest_path = os.path.join(ARGS.dataset, "manifest.json")
source = None
if os.path.isfile(manifest_path):
    with open(manifest_path) as fp:
        source = json.load(fp).get("source")
report = dict(dataset=ARGS.dataset, data_source=source, interval_s=D.dt, max_gap_s=ARGS.max_gap, horizon_s=ARGS.horizon,
              trips_total=len(D.files), minh=ARGS.minh, alert=dict(level=ARGS.alert_level, prob=ARGS.alert_prob, merge_s=ARGS.merge_s),
              settings={k: v for k, v in vars(ARGS).items() if k not in ("labels", "dataset", "out_dir")},
              reference="Full-horizon no-expiry pseudo-labels for all history settings",
              matrix_orientation="Rows are reference classes, columns are predicted classes. Binary order: negative, positive.",
              versions=dict(python=platform.python_version(), **{p: version(p) for p in
                            ("numpy", "pandas", "scipy", "scikit-learn", "xgboost")}))
report["cleaning"] = dict(trips=len(D.clean_report), raw_rows=sum(r["raw"] for r in D.clean_report.values()),
                          **{k: sum(r[k] for r in D.clean_report.values()) for k in
                             ("bad_rows", "duplicates", "poor_accuracy", "speed_over_50", "accel_over_10", "removed")},
                          unusable_trips=[f for f, r in D.clean_report.items() if not r["usable"]])
report["expiry_sweep"] = {n: summarize(RES[n][RES[n].trip.isin(common)], normal_files) if not RES[n].empty else {} for n in names}
report["common_trips"] = len(common)
report["base_evaluation"] = summarize(base, normal_files)
report["history_curve"] = []
report["by_direction"] = {}
if not base.empty:
    for lo, hi in [(ARGS.minh, 9), (max(ARGS.minh, 10), 19), (max(ARGS.minh, 20), 29),
                   (max(ARGS.minh, 30), 39), (max(ARGS.minh, 40), float("inf"))]:
        subset = base[base.n_hist.between(lo, hi)]
        if subset.trip.nunique() >= 3:
            report["history_curve"].append(dict(history=f"{lo}-{hi}" if np.isfinite(hi) else f"{lo}+",
                                                **summarize(subset, normal_files)))
    report["by_direction"] = {direction: summarize(base[base.file.str.startswith(prefix)], normal_files)
                              for direction, prefix in (("A to B", "a-to-b"), ("B to A", "b-to-a"))}
report["noted_events"] = dict(status="not available - no trip-notes file was given")
if NOTES is not None:
    report["noted_events"] = dict(status="reviewed notes", results={
        name: {model: field_check(frame, NOTES, alert) for model, alert in
               (("forecast", alert_forecast), ("current_level", alert_now), ("fixed_distance", alert_fixed))}
        for name, frame in RES.items()})
report["timing_s"] = {f"{name}_{stat}": float(func(v)) for name, v in TIMES.items() if v
                      for stat, func in (("median", np.median), ("max", np.max))}
report["timing_s"]["n"] = len(TIMES["fit"])
report["runtime_s"] = round(time.time() - t0, 1)

def json_ready(value):
    if isinstance(value, dict):
        return {k: json_ready(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_ready(v) for v in value]
    if isinstance(value, np.generic):
        return json_ready(value.item())
    return None if isinstance(value, float) and not np.isfinite(value) else value

report = json_ready(report)
os.makedirs(ARGS.out_dir, exist_ok=True)
with open(os.path.join(ARGS.out_dir, "evaluation_v4.json"), "w") as fp:
    json.dump(report, fp, indent=1, allow_nan=False)
base.to_csv(os.path.join(ARGS.out_dir, "replay_output_v4.csv"), index=False)

# ------------------------------------------------------------------ report (html)
f = lambda v, d=3: "-" if v is None or (isinstance(v, float) and np.isnan(v)) else (f"{v:.{d}f}" if isinstance(v, float) else str(v))
def table(head, rows): return "<table><tr>" + "".join(f"<th>{html.escape(h)}</th>" for h in head) + "</tr>" + "".join("<tr>" + "".join(f"<td>{html.escape(str(c))}</td>" for c in r) + "</tr>" for r in rows) + "</table>"
sec = []
if report["data_source"] == "synthetic":
    sec.append("<p><strong>Generated test data.</strong> Results describe a software replay only.</p>")
c = report["cleaning"]
sec.append("<h2>GPS cleaning</h2><p>" + html.escape(f"{c['trips']} trips, {c['raw_rows']} records. Removed {c['removed']}: speed over 50 m/s {c['speed_over_50']}, acceleration over 10 m/s2 {c['accel_over_10']}, error radius over 50 m {c['poor_accuracy']}, duplicates {c['duplicates']}, bad rows {c['bad_rows']}. Trips set aside as unusable: {len(c['unusable_trips'])}.") + "</p>")
keys = [("trips", "Trips judged"), ("ap_model_1", "Forecast AP (any concern)"), ("ap_persist_1", "Baseline AP (any concern)"), ("ap_model_2", "Forecast AP (Major+)"), ("ap_persist_2", "Baseline AP (Major+)"),
        ("mae_model", "Level error, forecast"), ("mae_persist", "Level error, baseline"), ("ece_1", "ECE (any concern)"), ("ece_2", "ECE (Major+)"),
        ("conf_coverage", "Empirical prediction-set coverage"), ("false_alarms_per_trip_forecast", "False alarms per verified ordinary trip, forecast"), ("false_alarms_per_trip_persist", "False alarms per verified ordinary trip, baseline")]
sec.append("<h2>History length (expiry) - same " + str(report["common_trips"]) + " trips for every row</h2>" + table(["History kept"] + [k[1] for k in keys], [[n] + [f(v.get(k)) for k, _ in keys] for n, v in report["expiry_sweep"].items()]))
sec.append("<h2>As the history grows (cold start), no expiry</h2>" + table(["Earlier trips in history"] + [k[1] for k in keys], [[r["history"]] + [f(r.get(k)) for k, _ in keys] for r in report["history_curve"]]))
def show_matrix(title, labels, matrix):
    return "<h3>" + html.escape(title) + "</h3>" + table(
        ["Reference / predicted", *labels], [[label, *row] for label, row in zip(labels, matrix)])

main = report["base_evaluation"]
sec.append("<h2>Full-window forecast confusion matrices</h2><p>Reference: no-expiry model-defined concern.</p>")
if main:
    sec.append(f"<p>{main['scored_windows']} scored windows, {main['unscored_windows']} unscored.</p>")
    for key, name in (("forecast", "Forecast"), ("persistence", "Persistence"), ("fixed_distance", "Fixed-distance rule")):
        values = main["alert_matrices"][str(ARGS.alert_level)][key]
        sec.append(show_matrix(name, ["Below threshold", "At or above threshold"], values["matrix"]))
        sec.append(f"<p>Precision {f(values['precision'])}, recall {f(values['recall'])}.</p>")
    onset = main["alert_matrices"][str(ARGS.alert_level)]["forecast_from_below_threshold"]
    sec.append(show_matrix("Forecast from below the alert level", ["No escalation", "Escalation"], onset["matrix"]))
    sec.append(show_matrix("Forecast by concern level", M.LEVELS, main["ordinal_matrix"]))
else:
    sec.append("<p>No trips had enough prior usable history.</p>")
ne = report["noted_events"]
sec.append("<h2>Independently reviewed trip events</h2>")
if NOTES is None:
    sec.append("<p>Unavailable. No reviewed trip notes were supplied.</p>")
else:
    for name, values in ne["results"].get("none", {}).items():
        if values["scored_trips"]:
            sec.append(show_matrix(name.replace("_", " ").title(), ["No event", "Event"], values["trip_matrix"]["matrix"]))
        else:
            sec.append(f"<p>{name.replace('_', ' ').title()} has no eligible reviewed trips.</p>")
        sec.append(f"<p>Early {values['early_events']}, late {values['late_events']}, missed {values['missed_events']}, "
                   f"unscored {values['unscored_events']}. Median lead {f(values['median_lead_s'])} seconds.</p>")
interval = f(D.dt)
with open(os.path.join(ARGS.out_dir, "evaluation_v4.html"), "w") as fp:
    fp.write("<!doctype html><meta charset=utf-8><title>Evaluation</title>"
             "<style>body{font:14px Georgia;max-width:1100px;margin:2em auto}table{border-collapse:collapse;margin:1em 0}"
             "td,th{border:1px solid #bbb;padding:4px 8px;text-align:left}</style><h1>LIGTAS replay evaluation</h1><p>"
             + html.escape(f"Dataset {ARGS.dataset}. Interval {interval} s. Runtime {report['runtime_s']} s.")
             + "</p>" + "".join(sec))
print(json.dumps({k: report[k] for k in ("cleaning", "base_evaluation", "noted_events")}, indent=1))
