"""
LIGTAS-CASCADE prototype
Context-conditioned Isolation Forest  ->  forward-risk XGBoost  ->  Platt-calibrated Safety Index
Inputs: only timestamp, latitude, longitude (route geometry is used for map-matching).
Run:  python3 pipeline.py <dataset_dir> <out_json>
"""
import json, sys, warnings
import numpy as np, pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss
import xgboost as xgb

warnings.filterwarnings("ignore")
DS = sys.argv[1]; OUT = sys.argv[2]
H = 6            # forecast horizon: 6 samples x 10 s = 60 s
Q_EVENT = 0.90   # an "anomalous moment" = IF score above the 90th percentile of training scores
SEG = 100.0      # metres per along-route bin
SEED = 7

man = json.load(open(f"{DS}/manifest.json"))
proj = json.load(open(f"{DS}/project.json"))["routes"][0]
LAT0, LON0 = 18.056, 120.556
KX, KY = 111320 * np.cos(np.radians(LAT0)), 110574.0

def to_xy(lat, lon):
    return (np.asarray(lon) - LON0) * KX, (np.asarray(lat) - LAT0) * KY

def polyline(coords):
    c = np.array(coords); x, y = to_xy(c[:, 1], c[:, 0])
    P = np.c_[x, y]; seg = np.hypot(*np.diff(P, axis=0).T)
    return P, np.r_[0, np.cumsum(seg)]

PATH = {"outbound": polyline(proj["paths"]["outbound"]["coords"]),
        "return": polyline(proj["paths"]["return"]["coords"])}

def project(P, cum, px, py):
    """nearest-point projection -> (along-route s, cross-track d)"""
    s_out = np.zeros(len(px)); d_out = np.zeros(len(px))
    A, B = P[:-1], P[1:]; AB = B - A; L2 = (AB ** 2).sum(1) + 1e-9
    for i, (x, y) in enumerate(zip(px, py)):
        t = np.clip(((x - A[:, 0]) * AB[:, 0] + (y - A[:, 1]) * AB[:, 1]) / L2, 0, 1)
        cx, cy = A[:, 0] + t * AB[:, 0], A[:, 1] + t * AB[:, 1]
        dd = np.hypot(x - cx, y - cy); k = dd.argmin()
        s_out[i] = cum[k] + t[k] * np.sqrt(L2[k]); d_out[i] = dd[k]
    return s_out, d_out

# ------------------------------------------------------------------ 1. kinematics
def trip_frame(rec, tid):
    df = pd.read_csv(f"{DS}/{rec['file']}")
    t = pd.to_datetime(df.timestamp, utc=True)
    x, y = to_xy(df.latitude, df.longitude)
    dt = t.diff().dt.total_seconds().values; dt[0] = 10.0
    step = np.r_[0, np.hypot(np.diff(x), np.diff(y))]
    v = step / dt                                   # m/s
    a = np.r_[0, np.diff(v)] / dt                   # m/s^2
    jerk = np.r_[0, np.diff(a)] / dt
    hd = np.arctan2(np.r_[0, np.diff(y)], np.r_[0, np.diff(x)])
    turn = np.abs(np.r_[0, (np.diff(np.unwrap(hd)))]) / dt   # rad/s
    turn[(v < 0.5)] = 0                             # heading is meaningless when stationary
    key = "outbound" if rec["direction"] == "outbound" else "return"
    P, cum = PATH[key]
    s, d = project(P, cum, x, y)
    if key == "return": s = cum[-1] - s             # express every trip on one A->B axis
    local = t.dt.tz_convert("Asia/Manila")
    f = pd.DataFrame(dict(
        trip=tid, dir=int(key == "return"), t=t, v=v, a=a, jerk=jerk, turn=turn, s=s, d=d,
        hour=local.dt.hour + local.dt.minute / 60, dow=local.dt.dayofweek,
        elapsed=(t - t.iloc[0]).dt.total_seconds().values))
    f["stopped"] = (f.v < 0.5).astype(int)
    # strictly backward-looking rolling context (window = 6 samples = 60 s)
    f["v_mean6"] = f.v.rolling(6, min_periods=1).mean()
    f["v_std6"] = f.v.rolling(6, min_periods=2).std().fillna(0)
    f["a_min6"] = f.a.rolling(6, min_periods=1).min()
    f["a_absmax6"] = f.a.abs().rolling(6, min_periods=1).max()
    f["stops6"] = f.stopped.rolling(6, min_periods=1).sum()
    run = np.zeros(len(f)); c = 0
    for i, z in enumerate(f.stopped):
        c = c + 1 if z else 0; run[i] = c
    f["dwell"] = run * 10                           # seconds currently stopped
    f["turn_sum6"] = f.turn.rolling(6, min_periods=1).sum()
    f["sbin"] = (f.s // SEG).astype(int).clip(lower=0)
    return f

recs = man["recordings"]
tripmeta = {t["id"]: t for t in proj["trips"]}
frames = []
for i, r in enumerate(recs):
    frames.append(trip_frame(r, i))
D = pd.concat(frames, ignore_index=True)
dep = pd.Series({i: pd.to_datetime(r["departure"]) for i, r in enumerate(recs)}).sort_values()
order = list(dep.index)                          # chronological trip order
NB = int(D.sbin.max()) + 1

# ------------------------------------------------------------------ 2. Stage A: context-conditioned IF
IF_FEATS = ["v", "a", "jerk", "turn", "d", "v_resid", "v_std6", "dwell"]

def stage_A(train_ids):
    tr = D[D.trip.isin(train_ids)]
    # contextual baseline: median speed per (along-route bin x direction x half-day) on TRAIN trips only
    D["part"] = (D.hour >= 12).astype(int)
    base = tr.assign(part=(tr.hour >= 12).astype(int)).groupby(["sbin", "dir", "part"]).v.median()
    gmed = tr.v.median()
    key = list(zip(D.sbin, D.dir, D.part))
    D["v_base"] = [base.get(k, gmed) for k in key]
    D["v_resid"] = D.v - D.v_base
    tr = D[D.trip.isin(train_ids)]
    iso = IsolationForest(n_estimators=300, max_samples=256, contamination="auto", random_state=SEED)
    iso.fit(tr[IF_FEATS])
    raw = -iso.score_samples(D[IF_FEATS])         # higher = more isolated
    D["if_score"] = raw
    tau = np.quantile(raw[D.trip.isin(train_ids)], Q_EVENT)
    D["event"] = (raw >= tau).astype(int)
    return iso, tau

# ------------------------------------------------------------------ 3. targets
def make_targets():
    y = np.zeros(len(D), int); y_on = np.full(len(D), np.nan)
    for tid, g in D.groupby("trip", sort=False):
        e = g.event.values; n = len(e); idx = g.index.values
        fut = np.array([e[i + 1:i + 1 + H].max() if i + 1 < n else 0 for i in range(n)])
        y[idx] = fut
        y_on[idx] = np.where(e == 0, fut, np.nan)      # onset: only moments that are normal NOW
    D["y"] = y; D["y_onset"] = y_on

# ------------------------------------------------------------------ 4. segment memory (out-of-trip)
def seg_prior(train_ids, alpha=2.0):
    tr = D[D.trip.isin(train_ids)]
    g_rate = tr.y.mean()
    cnt = tr.groupby("sbin").y.agg(["sum", "count"]).reindex(range(NB), fill_value=0)
    prior = np.zeros(len(D))
    for i, (tid, sb, yy) in enumerate(zip(D.trip.values, D.sbin.values, D.y.values)):
        sm, ct = cnt.loc[sb, "sum"], cnt.loc[sb, "count"]
        if tid in train_ids:                            # leave-own-trip-out for training rows
            own = D[(D.trip == tid) & (D.sbin == sb)]
            sm -= own.y.sum(); ct -= len(own)
        prior[i] = (sm + alpha * g_rate) / (ct + alpha)
    D["seg_prior"] = prior

BASE_FEATS = ["v", "a", "jerk", "turn", "d", "v_mean6", "v_std6", "a_min6", "a_absmax6", "stops6",
              "dwell", "turn_sum6", "s", "dir", "hour", "dow", "elapsed", "v_resid"]
FULL_FEATS = BASE_FEATS + ["if_score", "seg_prior"]

def ece(y, p, bins=8):
    q = np.quantile(p, np.linspace(0, 1, bins + 1)); q[-1] += 1e-9; e = 0
    for lo, hi in zip(q[:-1], q[1:]):
        m = (p >= lo) & (p < hi)
        if m.sum(): e += m.mean() * abs(y[m].mean() - p[m].mean())
    return e

def fit_xgb(Xtr, ytr, Xva, yva):
    spw = 1.0   # no re-weighting: keeps probabilities near-calibrated (cf. Niculescu-Mizil & Caruana 2005)
    m = xgb.XGBClassifier(n_estimators=400, learning_rate=0.04, max_depth=3, subsample=0.8,
                          colsample_bytree=0.8, min_child_weight=3, reg_lambda=2.0,
                          scale_pos_weight=spw, eval_metric="aucpr", early_stopping_rounds=40,
                          random_state=SEED, n_jobs=2)
    m.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
    return m

def logit(p): p = np.clip(p, 1e-4, 1 - 1e-4); return np.log(p / (1 - p)).reshape(-1, 1)

def run(train_ids, val_ids, test_ids, want_detail=False):
    allfit = list(train_ids)
    iso, tau = stage_A(allfit)
    make_targets()
    seg_prior(set(allfit))
    tr, va, te = (D[D.trip.isin(s)] for s in (train_ids, val_ids, test_ids))
    res = {}
    # baselines
    res["IF only (current score)"] = te.if_score.values
    m_base = fit_xgb(tr[BASE_FEATS], tr.y, va[BASE_FEATS], va.y)
    res["XGBoost, no IF"] = m_base.predict_proba(te[BASE_FEATS])[:, 1]
    F_noseg = BASE_FEATS + ["if_score"]
    m_ns = fit_xgb(tr[F_noseg], tr.y, va[F_noseg], va.y)
    res["XGBoost + IF score (no segment memory)"] = m_ns.predict_proba(te[F_noseg])[:, 1]
    F_noif = BASE_FEATS + ["seg_prior"]
    m_ni = fit_xgb(tr[F_noif], tr.y, va[F_noif], va.y)
    res["XGBoost + segment memory (no IF score)"] = m_ni.predict_proba(te[F_noif])[:, 1]
    m_if = fit_xgb(tr[FULL_FEATS], tr.y, va[FULL_FEATS], va.y)
    raw_va = m_if.predict_proba(va[FULL_FEATS])[:, 1]
    res["LIGTAS-CASCADE (raw)"] = m_if.predict_proba(te[FULL_FEATS])[:, 1]
    cal = LogisticRegression(C=1e3).fit(logit(raw_va), va.y)
    p_cal = cal.predict_proba(logit(res["LIGTAS-CASCADE (raw)"]))[:, 1]
    res["LIGTAS-CASCADE (calibrated)"] = p_cal
    y = te.y.values; yo = te.y_onset.values; mo = ~np.isnan(yo)
    rows = {}
    for k, p in res.items():
        r = dict(prauc=average_precision_score(y, p), roc=roc_auc_score(y, p))
        r["prauc_onset"] = average_precision_score(yo[mo], p[mo]) if yo[mo].sum() > 0 else np.nan
        r["roc_onset"] = roc_auc_score(yo[mo], p[mo]) if yo[mo].sum() > 0 else np.nan
        if "CASCADE" in k or "XGBoost" in k:
            r["brier"] = brier_score_loss(y, np.clip(p, 0, 1)); r["ece"] = ece(y, np.clip(p, 0, 1))
        rows[k] = r
    out = dict(rows=rows, base_rate=float(y.mean()), n_test=int(len(te)), n_pos=int(y.sum()),
               onset_n=int(mo.sum()), onset_pos=int(np.nansum(yo)), tau=float(tau))
    if want_detail:
        out.update(dict(model=m_if, cal=cal, te=te.copy(), p_cal=p_cal, raw_te=res["LIGTAS-CASCADE (raw)"],
                        tr=tr, va=va, m_base=m_base))
    return out

# ------------------------------------------------------------------ 5. main split (chronological, by trip)
n = len(order); a, b = int(n * .70), int(n * .85)
tr_ids, va_ids, te_ids = order[:a], order[a:b], order[b:]
main = run(tr_ids, va_ids, te_ids, want_detail=True)

# ------------------------------------------------------------------ 6. rolling-origin CV (expanding window)
cv = []
for k in (24, 30, 36, 42, 48):
    trk = order[:k]; vak = order[k - 6:k]; tek = order[k:k + 6 if k + 6 <= n else n]
    if len(tek) < 3: continue
    # validation = last 6 of the training span (early stopping + calibration), fit on the rest
    r = run(order[:k - 6], vak, tek)
    cv.append(dict(train_trips=k - 6, test_trips=len(tek), **{kk: v for kk, v in r["rows"].items()},
                   base_rate=r["base_rate"]))
cvsum = {}
for name in main["rows"]:
    cvsum[name] = {m: float(np.nanmean([c[name][m] for c in cv if m in c[name]]))
                   for m in ("prauc", "roc", "prauc_onset", "roc_onset") }
    cvsum[name]["prauc_sd"] = float(np.nanstd([c[name]["prauc"] for c in cv]))
    cvsum[name]["roc_sd"] = float(np.nanstd([c[name]["roc"] for c in cv]))

# ------------------------------------------------------------------ 7. explanations + exports (main split)
run_main = main
te = run_main["te"].copy(); te["p_cal"] = run_main["p_cal"]; te["p_raw"] = run_main["raw_te"]
mdl = run_main["model"]
contrib = mdl.get_booster().predict(xgb.DMatrix(te[FULL_FEATS]), pred_contribs=True)[:, :-1]
shap_abs = dict(zip(FULL_FEATS, np.abs(contrib).mean(0).round(4)))
gain = mdl.get_booster().get_score(importance_type="total_gain")
gsum = sum(gain.values())
gain = {k: round(v / gsum, 4) for k, v in gain.items()}

# refit the pipeline on ALL trips only for the route-risk map (descriptive, clearly not a test result)
iso_all, tau_all = stage_A(order)
make_targets(); seg_prior(set(order))
route_bins = []
for sb in range(NB):
    g = D[D.sbin == sb]
    if len(g) == 0: continue
    route_bins.append(dict(bin=int(sb), s0=int(sb * SEG), n=int(len(g)),
                           event_rate=round(float(g.event.mean()), 3),
                           fwd_rate=round(float(g.y.mean()), 3),
                           speed_kmh=round(float(g.v.median() * 3.6), 1),
                           stopped=round(float(g.stopped.mean()), 3)))
# geometry for plotting (outbound polyline lon/lat + cumulative distance)
ob = np.array(proj["paths"]["outbound"]["coords"]); _, cum_ob = PATH["outbound"]

# pick a test trip with positives for the timeline demo
cand = te.groupby("trip").y.sum().sort_values(ascending=False)
demo_trip = int(cand.index[0]); dm = te[te.trip == demo_trip]
demo = dict(trip=demo_trip, direction="B→A" if dm.dir.iloc[0] else "A→B",
            start=str(dm.t.iloc[0].tz_convert("Asia/Manila"))[:16],
            t=[int(x) for x in dm.elapsed], speed_kmh=[round(float(x * 3.6), 1) for x in dm.v],
            if_score=[round(float(x), 4) for x in dm.if_score], p=[round(float(x), 4) for x in dm.p_raw],
            y=[int(x) for x in dm.y], event=[int(x) for x in dm.event],
            safety=[round(100 * (1 - float(x)), 1) for x in dm.p_raw])

# reliability diagram on test
pc = np.clip(te.p_cal.values, 0, 1); yy = te.y.values
qs = np.quantile(pc, np.linspace(0, 1, 6)); qs[-1] += 1e-9; rel = []
for lo, hi in zip(qs[:-1], qs[1:]):
    m = (pc >= lo) & (pc < hi)
    if m.sum(): rel.append(dict(pred=round(float(pc[m].mean()), 3), obs=round(float(yy[m].mean()), 3), n=int(m.sum())))
pr_hat = te.p_raw.values
rel_raw = []
qs2 = np.quantile(pr_hat, np.linspace(0, 1, 6)); qs2[-1] += 1e-9
for lo, hi in zip(qs2[:-1], qs2[1:]):
    m = (pr_hat >= lo) & (pr_hat < hi)
    if m.sum(): rel_raw.append(dict(pred=round(float(pr_hat[m].mean()), 3), obs=round(float(yy[m].mean()), 3), n=int(m.sum())))

# data summary
summary = dict(
    n_trips=len(recs), n_points=int(len(D)), sampling_s=10,
    n_train=len(tr_ids), n_val=len(va_ids), n_test=len(te_ids),
    train_period=[str(dep.iloc[0])[:10], str(dep[tr_ids[-1]])[:10]],
    test_period=[str(dep[te_ids[0]])[:10], str(dep[te_ids[-1]])[:10]],
    speed_p50_kmh=round(float(D.v.median() * 3.6), 1), speed_p95_kmh=round(float(D.v.quantile(.95) * 3.6), 1),
    accel_min=round(float(D.a.min()), 2), accel_max=round(float(D.a.max()), 2),
    stopped_frac=round(float(D.stopped.mean()), 3), route_len_m=round(float(cum_ob[-1])),
    dir_counts={"A→B": int((pd.Series([r["direction"] for r in recs]) == "outbound").sum()),
                "B→A": int((pd.Series([r["direction"] for r in recs]) != "outbound").sum())},
    vehicles=sorted(set(t["vehicle"] for t in proj["trips"])),
    hours=sorted(set(int(h) for h in D.groupby("trip").hour.first())),
    max_cross_track_m=round(float(D.d.max()), 1))

result = dict(
    summary=summary, params=dict(H=H, Q_EVENT=Q_EVENT, SEG=SEG, tau=main["tau"],
                                 IF_FEATS=IF_FEATS, BASE_FEATS=BASE_FEATS, FULL_FEATS=FULL_FEATS),
    main={k: v for k, v in main.items() if k in ("rows", "base_rate", "n_test", "n_pos", "onset_n", "onset_pos")},
    cv=cv, cvsum=cvsum, shap_abs=shap_abs, gain=gain, route_bins=route_bins, demo=demo,
    rel=rel, rel_raw=rel_raw,
    geometry=dict(lon=[round(float(x), 6) for x in ob[:, 0]], lat=[round(float(x), 6) for x in ob[:, 1]],
                  cum=[round(float(x), 1) for x in cum_ob]))
json.dump(result, open(OUT, "w"), indent=1, default=float)
print(json.dumps(dict(summary=summary, main=result["main"], cvsum=cvsum), indent=1, default=float))
