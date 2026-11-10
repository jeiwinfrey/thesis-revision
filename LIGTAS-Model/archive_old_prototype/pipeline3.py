"""
LIGTAS-CASCADE v3
 - time-aware, shrinkage baseline (hour-block x road stretch x direction)
 - distribution-based speed features + run-up features (weak-spot fixes)
 - ordinal "concern level" (Normal / Minor / Major / Critical) via cumulative binary XGBoost (Frank & Hall 2001)
 - calibration (raw / Platt / isotonic, evaluated on earlier->later folds)
 - uncertainty: trip-bootstrap ensemble spread, split-conformal and adaptive conformal prediction sets
 - injected-anomaly benchmark incl. run-up and gradual-drift events
Run: python3 pipeline3.py <dataset_dir> <out_json> <sample_csv>
"""
import json, sys, warnings, math
import numpy as np, pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss, log_loss, cohen_kappa_score
import xgboost as xgb

warnings.filterwarnings("ignore")
DS, OUT, SAMPLE = sys.argv[1], sys.argv[2], sys.argv[3]
H, SEG, SEED = 6, 100.0, 7
BANDS = [150.0, 500.0, 1000.0]
DWELL_LV = [30.0, 90.0, 180.0]
IF_Q = [0.90, 0.975, 0.995]
SHRINK_K = 8.0
LEVELS = ["Normal", "Minor", "Major", "Critical"]

man = json.load(open(f"{DS}/manifest.json"))
proj = json.load(open(f"{DS}/project.json"))["routes"][0]
LAT0, LON0 = 18.056, 120.556
KX, KY = 111320 * np.cos(np.radians(LAT0)), 110574.0
to_xy = lambda lat, lon: ((np.asarray(lon) - LON0) * KX, (np.asarray(lat) - LAT0) * KY)

def polyline(coords):
    c = np.array(coords); x, y = to_xy(c[:, 1], c[:, 0]); P = np.c_[x, y]
    return P, np.r_[0, np.cumsum(np.hypot(*np.diff(P, axis=0).T))]
PATH = {"outbound": polyline(proj["paths"]["outbound"]["coords"]), "return": polyline(proj["paths"]["return"]["coords"])}

def project(P, cum, px, py):
    A, B = P[:-1], P[1:]; AB = B - A; L2 = (AB ** 2).sum(1) + 1e-9
    s = np.zeros(len(px)); d = np.zeros(len(px)); brg = np.zeros(len(px))
    for i, (x, y) in enumerate(zip(px, py)):
        t = np.clip(((x - A[:, 0]) * AB[:, 0] + (y - A[:, 1]) * AB[:, 1]) / L2, 0, 1)
        dd = np.hypot(x - (A[:, 0] + t * AB[:, 0]), y - (A[:, 1] + t * AB[:, 1])); k = dd.argmin()
        s[i] = cum[k] + t[k] * np.sqrt(L2[k]); d[i] = dd[k]; brg[i] = math.atan2(AB[k, 1], AB[k, 0])
    return s, d, brg
wrap = lambda a: (a + np.pi) % (2 * np.pi) - np.pi

def hour_block(h):  # 0 night, 1 am peak, 2 midday, 3 pm peak
    return np.where((h < 6) | (h >= 19), 0, np.where(h < 10, 1, np.where(h < 14, 2, 3)))

# ------------------------------------------------------------------ frames (training-independent features)
def make_frame(tid, t, x, y, key):
    dt = np.r_[10.0, np.diff(t.asi8 / 1e9)]
    step = np.r_[0, np.hypot(np.diff(x), np.diff(y))]
    v = step / dt; a = np.r_[0, np.diff(v)] / dt; jerk = np.r_[0, np.diff(a)] / dt
    hd = np.arctan2(np.r_[0, np.diff(y)], np.r_[0, np.diff(x)])
    turn = np.abs(np.r_[0, wrap(np.diff(hd))]) / dt; turn[v < 0.5] = 0
    P, cum = PATH[key]; s, d, brg = project(P, cum, x, y)
    hdev = np.degrees(np.abs(wrap(hd - brg))); hdev[v < 1.0] = 0
    if key == "return": s = cum[-1] - s
    loc = t.tz_convert("Asia/Manila")
    f = pd.DataFrame(dict(trip=tid, dir=int(key == "return"), t=t, x=x, y=y, v=v, a=a, jerk=jerk, turn=turn, s=s, d=d,
                          heading_dev=hdev, hour=loc.hour + loc.minute / 60, dow=loc.dayofweek,
                          elapsed=(t - t[0]).total_seconds().values))
    f["tb"] = hour_block(f.hour.values)
    f["stopped"] = (f.v < 0.5).astype(int)
    f["v_mean6"] = f.v.rolling(6, min_periods=1).mean()
    f["v_std6"] = f.v.rolling(6, min_periods=2).std().fillna(0)
    f["a_min6"] = f.a.rolling(6, min_periods=1).min()
    f["a_absmax6"] = f.a.abs().rolling(6, min_periods=1).max()
    f["stops6"] = f.stopped.rolling(6, min_periods=1).sum()
    f["turn_sum6"] = f.turn.rolling(6, min_periods=1).sum()
    run = np.zeros(len(f)); c = 0; tss = np.zeros(len(f)); cc = 600
    for i, z in enumerate(f.stopped):
        c = c + 1 if z else 0; run[i] = c
        cc = 0 if z else min(cc + 10, 600); tss[i] = cc
    f["dwell"] = run * 10; f["time_since_stop"] = tss
    f["cum_stop"] = f.stopped.cumsum() * 10
    f["sbin"] = (f.s // SEG).astype(int).clip(lower=0)
    # run-up and rate-of-change features (backward-looking only)
    f["v_trend3"] = ((f.v - f.v.shift(3)) / 30).fillna(0)
    f["a_mean3"] = f.a.rolling(3, min_periods=1).mean()
    f["d_rate3"] = ((f.d - f.d.shift(3)) / 30).fillna(0)
    f["hdev_rate"] = f.heading_dev.diff().fillna(0)
    mv = f.v.where(f.v > 0.5); f["v_run"] = mv.expanding().mean().fillna(0)
    f["v_rel"] = f.v / (f.v_run + 1.0)
    return f

recs = man["recordings"]; RAW = {}
for i, r in enumerate(recs):
    df = pd.read_csv(f"{DS}/{r['file']}"); t = pd.DatetimeIndex(pd.to_datetime(df.timestamp, utc=True))
    x, y = to_xy(df.latitude, df.longitude)
    RAW[i] = dict(t=t, x=np.asarray(x), y=np.asarray(y), key="outbound" if r["direction"] == "outbound" else "return")
FR = {i: make_frame(i, v["t"], v["x"], v["y"], v["key"]) for i, v in RAW.items()}
dep = pd.Series({i: pd.to_datetime(r["departure"]) for i, r in enumerate(recs)}).sort_values()
order = list(dep.index)

# ------------------------------------------------------------------ context (training-dependent)
class Ctx: pass
def mad(a): a = np.asarray(a); return float(np.median(np.abs(a - np.median(a))) * 1.4826)

def fit_ctx(train_ids, mode="time"):
    Ft = pd.concat([FR[i] for i in train_ids], ignore_index=True); c = Ctx(); c.mode = mode; c.train = set(train_ids)
    mv = Ft[Ft.v > 0.5]; c.gm, c.gs = float(mv.v.median()), max(mad(mv.v), 1.0)
    def table(keys):
        g = mv.groupby(keys).v; return {k: (float(m), max(float(s), 1.0), int(n)) for k, m, s, n in zip(g.median().index, g.median(), g.apply(mad), g.size())}
    c.seg = table(["sbin", "dir"]); c.tb = table(["sbin", "dir", "tb"])
    mv2 = mv.assign(part=(mv.hour >= 12).astype(int)); g = mv2.groupby(["sbin", "dir", "part"]).v
    c.ampm = {k: (float(m), max(float(s), 1.0), int(n)) for k, m, s, n in zip(g.median().index, g.median(), g.apply(mad), g.size())}
    c.dref = Ft.groupby(["sbin", "dir"]).dwell.quantile(0.9).to_dict()
    c.pstop = Ft.groupby(["sbin", "dir"]).stopped.mean().to_dict()
    c.zone = {dr: np.array(sorted(b for (b, d_), p in c.pstop.items() if d_ == dr and p >= 0.25)) for dr in (0, 1)}
    c.hist = {}
    for dr in (0, 1):
        g2 = Ft[Ft.dir == dr]; c.hist[dr] = (np.c_[g2.x, g2.y], g2.trip.values)
    return c

def baseline(F, c):
    est = np.full(len(F), c.gm); sig = np.full(len(F), c.gs)
    if c.mode == "none": return est, sig
    for i, (sb, dr, tb, h) in enumerate(zip(F.sbin.values, F.dir.values, F.tb.values, F.hour.values)):
        m0 = c.seg.get((sb, dr)); l1 = c.tb.get((sb, dr, tb)) if c.mode == "time" else c.ampm.get((sb, dr, int(h >= 12)))
        if m0 is None: continue
        if l1 is None: est[i], sig[i] = m0[0], m0[1]
        else:
            w = l1[2] / (l1[2] + SHRINK_K); est[i] = w * l1[0] + (1 - w) * m0[0]; sig[i] = w * l1[1] + (1 - w) * m0[1]
    return est, sig

def apply_ctx(F, c):
    F = F.reset_index(drop=True).copy()
    est, sig = baseline(F, c); F["v_base"] = est; F["v_resid"] = F.v - est
    F["v_z"] = np.clip((F.v - est) / sig, -6, 6)
    F["v_z_mean5"] = F.groupby("trip").v_z.transform(lambda s: s.rolling(5, min_periods=1).mean())
    F["dwell_ref"] = [c.dref.get(k, 0.0) for k in zip(F.sbin, F.dir)]
    F["excess_dwell"] = (F.dwell - F.dwell_ref).clip(lower=0)
    F["stop_unexp"] = F.stopped * (1 - np.array([c.pstop.get(k, 0.0) for k in zip(F.sbin, F.dir)]))
    dh = np.zeros(len(F))
    for tid, g in F.groupby("trip"):
        Hp, ht = c.hist[int(g.dir.iloc[0])]; m = ht != tid; P = np.c_[g.x, g.y]
        dh[g.index] = np.sqrt(((P[:, None, :] - Hp[m][None, :, :]) ** 2).sum(2)).min(1) if m.any() else 9e3
    F["d_hist"] = dh
    F["band_hist"] = np.digitize(F.d_hist, BANDS, right=True); F["band_poly"] = np.digitize(F.d, BANDS, right=True)
    dr_ = np.zeros(len(F))
    for tid, g in F.groupby("trip"):
        run = 0
        for i, b in zip(g.index, np.maximum(g.band_hist, g.band_poly).values):
            run = run + 10 if b >= 1 else 0; dr_[i] = run
    F["dev_run"] = dr_
    dz = np.full(len(F), 600.0)
    for i, (s, dr) in enumerate(zip(F.s.values, F.dir.values)):
        z = c.zone[int(dr)]
        if len(z):
            delta = ((z * SEG + SEG / 2) - s) * (1 if dr == 0 else -1); ah = delta[delta >= -20]
            if len(ah): dz[i] = min(ah.min(), 600.0)
    F["dist_to_stop_zone"] = np.maximum(dz, 0)
    return F

V1 = ["v", "a", "jerk", "turn", "d", "v_resid", "v_std6", "dwell"]
V2 = V1 + ["heading_dev", "excess_dwell", "d_hist"]
V3 = V2 + ["v_z", "v_z_mean5", "v_trend3", "d_rate3", "time_since_stop"]
BASE = ["v", "a", "jerk", "turn", "d", "v_mean6", "v_std6", "a_min6", "a_absmax6", "stops6", "dwell", "turn_sum6",
        "s", "dir", "hour", "dow", "elapsed", "v_resid"]
DEVG = ["d_hist", "band_hist", "band_poly", "heading_dev", "dev_run"]
DWG = ["excess_dwell", "stop_unexp", "cum_stop"]
RUN = ["v_trend3", "a_mean3", "time_since_stop", "dist_to_stop_zone", "d_rate3", "hdev_rate", "v_z", "v_z_mean5", "v_rel"]
F_V2 = BASE + DEVG + DWG
F_V3 = BASE + DEVG + DWG + RUN
F_ORD = F_V3 + ["lvl_now", "if3"]   # adds the forest/level state at time t (known at prediction time)

def levels(F, taus):
    ifl = np.digitize(F.if3.values, taus)
    bl = np.maximum(F.band_hist.values, F.band_poly.values)
    dl = np.digitize(F.excess_dwell.values, DWELL_LV)
    return np.maximum.reduce([ifl, bl, dl]), ifl

def fwd_max(F, col):
    out = np.zeros(len(F), int)
    for tid, g in F.groupby("trip", sort=False):
        e = g[col].values; n = len(e)
        out[g.index.values] = [e[i + 1:i + 1 + H].max() if i + 1 < n else 0 for i in range(n)]
    return out

def fit_xgb(Xtr, ytr, Xva, yva, seed=SEED):
    kw = dict(n_estimators=400, learning_rate=0.04, max_depth=3, subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
              reg_lambda=2.0, random_state=seed, n_jobs=2)
    if yva.sum() > 0 and ytr.sum() >= 3:
        m = xgb.XGBClassifier(eval_metric="aucpr", early_stopping_rounds=40, **kw); m.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
    elif ytr.sum() >= 3:
        kw["n_estimators"] = 150; m = xgb.XGBClassifier(**kw); m.fit(Xtr, ytr)
    else: return None
    return m
def pp(m, X, default=0.0): return m.predict_proba(X)[:, 1] if m is not None else np.full(len(X), default)
def safe_pr(y, p): return float(average_precision_score(y, p)) if y.sum() > 0 else np.nan
def safe_roc(y, p): return float(roc_auc_score(y, p)) if 0 < y.sum() < len(y) else np.nan
def tpr_at_fpr(neg, pos, fpr=0.05): return float((pos > np.quantile(neg, 1 - fpr)).mean()) if len(pos) else np.nan
def ecdf(ref): ref = np.sort(ref); return lambda s: np.searchsorted(ref, s, side="right") / len(ref)

# ------------------------------------------------------------------ injections
def inject_detour(raw, peak, rng, drift=False):
    t, x, y = raw["t"], raw["x"].copy(), raw["y"].copy(); n = len(x)
    r = 10 if drift else int(max(3, math.ceil(peak / 150))); L = 2 * r + 1
    if n < L + 10: return None
    i0 = int(rng.randint(4, n - L - 4)); hd = np.arctan2(y[i0 + r] - y[i0], x[i0 + r] - x[i0])
    nx, ny = -math.sin(hd), math.cos(hd); sg = rng.choice([-1, 1]); off = np.zeros(n)
    for j in range(L):
        off[i0 + j] = peak * (1 - abs(j - r) / (r + 1)); x[i0 + j] += sg * off[i0 + j] * nx; y[i0 + j] += sg * off[i0 + j] * ny
    mask = (off > 150) if drift else (off > 0)
    ws = int(np.argmax(mask)); return dict(t=t, x=x, y=y, key=raw["key"]), mask, ws

def inject_stop(raw, F_clean, k, rng, c, runup=False):
    n = len(raw["x"]); ps = np.array([c.pstop.get((int(b), int(F_clean.dir.iloc[0])), 0.0) for b in F_clean.sbin])
    cand = [i for i in range(7, n - 5) if ps[i] < 0.05] or list(range(7, n - 5)); i0 = int(rng.choice(cand))
    if not runup:
        x = np.r_[raw["x"][:i0 + 1], raw["x"][i0] + rng.randn(k) * 1.5, raw["x"][i0 + 1:]]
        y = np.r_[raw["y"][:i0 + 1], raw["y"][i0] + rng.randn(k) * 1.5, raw["y"][i0 + 1:]]; ws = i0 + 1
    else:
        c0 = np.r_[0, np.cumsum(np.hypot(np.diff(raw["x"]), np.diff(raw["y"])))] + np.arange(n) * 1e-6
        dl = np.diff(c0); w = np.linspace(0.8, 0.15, 5)
        dl2 = np.r_[dl[:i0 - 5], dl[i0 - 5:i0] * w, np.zeros(k), dl[i0:]]; c2 = np.r_[0, np.cumsum(dl2)]
        c2 = np.clip(c2, 0, c0[-1]); x = np.interp(c2, c0, raw["x"]); y = np.interp(c2, c0, raw["y"]); ws = i0 + 1
    t = pd.date_range(raw["t"][0], periods=len(x), freq="10s"); mask = np.zeros(len(x), bool); mask[ws:ws + k] = True
    return dict(t=t, x=x, y=y, key=raw["key"]), mask, ws

def inject_speed(raw, f, rng):
    n = len(raw["x"]); W = 8
    if n < W + 10: return None
    c0 = np.r_[0, np.cumsum(np.hypot(np.diff(raw["x"]), np.diff(raw["y"])))] + np.arange(n) * 1e-6
    i0 = int(rng.randint(3, n - W - 3)); tgt = c0.copy()
    tgt[i0:i0 + W] = c0[i0] + f * (c0[i0:i0 + W] - c0[i0]); tgt[i0 + W:] = c0[i0 + W:] + (f - 1) * (c0[i0 + W - 1] - c0[i0])
    tgt = np.clip(tgt, 0, c0[-1]); mask = np.zeros(n, bool); mask[i0:i0 + W] = True
    return dict(t=raw["t"], x=np.interp(tgt, c0, raw["x"]), y=np.interp(tgt, c0, raw["y"]), key=raw["key"]), mask, i0

KINDS = ["detour_minor", "detour_major", "detour_critical", "detour_drift_750", "stop_90s", "stop_180s", "stop_runup_90s", "speed_up", "crawl"]
EXPECT = {"detour_minor": 1, "detour_major": 2, "detour_critical": 3, "detour_drift_750": 2, "stop_90s": 1, "stop_180s": 2, "stop_runup_90s": 1, "speed_up": 1, "crawl": 1}

def build_injected(test_ids, c, seed):
    rng = np.random.RandomState(seed); out = []; uid = 10000 + seed * 1000
    for tid in test_ids:
        for kind in KINDS:
            if kind == "detour_minor": res = inject_detour(RAW[tid], 300., rng)
            elif kind == "detour_major": res = inject_detour(RAW[tid], 750., rng)
            elif kind == "detour_critical": res = inject_detour(RAW[tid], 1200., rng)
            elif kind == "detour_drift_750": res = inject_detour(RAW[tid], 750., rng, drift=True)
            elif kind == "stop_90s": res = inject_stop(RAW[tid], FR[tid], 9, rng, c)
            elif kind == "stop_180s": res = inject_stop(RAW[tid], FR[tid], 18, rng, c)
            elif kind == "stop_runup_90s": res = inject_stop(RAW[tid], FR[tid], 9, rng, c, runup=True)
            elif kind == "speed_up": res = inject_speed(RAW[tid], 1.5, rng)
            else: res = inject_speed(RAW[tid], 0.5, rng)
            if res is None: continue
            raw, mask, ws = res; uid += 1
            out.append((kind, tid, mask, ws, make_frame(uid, raw["t"], raw["x"], raw["y"], raw["key"])))
    return out

DET = ["if1", "if2", "if3", "d_hist", "excess_dwell", "vz", "xgb"]

# ------------------------------------------------------------------ one fold
def run_fold(tr_ids, va_ids, te_ids, mode="time", injection=False, light=False, seed=SEED):
    c = fit_ctx(tr_ids, mode)
    D = apply_ctx(pd.concat([FR[i] for i in tr_ids + va_ids + te_ids], ignore_index=True), c)
    tr = D.trip.isin(tr_ids).values; va = D.trip.isin(va_ids).values; te = D.trip.isin(te_ids).values
    mk = lambda cols: IsolationForest(n_estimators=300, max_samples=256, random_state=seed).fit(D.loc[tr, cols])
    if1, if2, if3 = mk(V1), mk(V2), mk(V3)
    D["if1"] = -if1.score_samples(D[V1]); D["if2"] = -if2.score_samples(D[V2]); D["if3"] = -if3.score_samples(D[V3])
    taus = np.quantile(D.loc[tr, "if3"], IF_Q)
    D["lvl_now"], ifl = levels(D, taus); D["ev_if"] = (ifl >= 1).astype(int)
    D["lvl_fwd"] = fwd_max(D, "lvl_now"); D["y_if"] = fwd_max(D, "ev_if")
    out = dict(c=c, D=D, taus=taus, te=te, tr=tr, va=va, if1=if1, if2=if2, if3=if3)
    if not light:
        y1 = (D.lvl_fwd >= 1).astype(int).values
        rows = {"Persistence (concern now → concern soon)": (safe_pr(y1[te], (D.lvl_now >= 1).values[te]), safe_roc(y1[te], (D.lvl_now >= 1).values[te]))}
        models = {}
        for gname, cols in (("v2 features", F_V2), ("v3 features (+run-up, time-aware)", F_V3), ("v3 + current concern level", F_ORD)):
            m = fit_xgb(D.loc[tr, cols], y1[tr], D.loc[va, cols], y1[va], seed); p = pp(m, D.loc[te, cols])
            rows[gname] = (safe_pr(y1[te], p), safe_roc(y1[te], p)); models[gname] = m
        yif = D.y_if.values
        rows_if = {"Persistence (concern now → concern soon)": (safe_pr(yif[te], D.ev_if.values[te]), safe_roc(yif[te], D.ev_if.values[te]))}
        m_if = fit_xgb(D.loc[tr, F_V3], yif[tr], D.loc[va, F_V3], yif[va], seed); p_if = pp(m_if, D.loc[te, F_V3])
        rows_if["v3 features (+run-up, time-aware)"] = (safe_pr(yif[te], p_if), safe_roc(yif[te], p_if))
        out["fore"] = {"Minor-or-worse in next 60 s": rows, "Forest-defined event in next 60 s": rows_if}
        out["base"] = {"Minor-or-worse in next 60 s": float(y1[te].mean()), "Forest-defined event in next 60 s": float(yif[te].mean())}
        main = models["v3 + current concern level"]; out["main"] = main
        # ordinal: cumulative binary models P(level >= k)
        ords = [main]
        for k in (2, 3):
            yk = (D.lvl_fwd >= k).astype(int).values; ords.append(fit_xgb(D.loc[tr, F_ORD], yk[tr], D.loc[va, F_ORD], yk[va], seed))
        P = np.column_stack([pp(m, D[F_ORD]) for m in ords]); P[:, 1] = np.minimum(P[:, 1], P[:, 0]); P[:, 2] = np.minimum(P[:, 2], P[:, 1])
        # decision thresholds tuned on the validation trips (maximise quadratic-weighted kappa); default 0.5 when too few positives
        thr = [0.5, 0.5, 0.5]; Yv = D.lvl_fwd.values[va]; Pv = P[va]
        for k in range(3):
            if (Yv >= k + 1).sum() >= 10:
                best, bt = -9, thr[k]
                for t_ in np.arange(0.10, 0.71, 0.05):
                    th = list(thr); th[k] = t_; pr_ = (Pv >= np.array(th)).sum(1); sc = cohen_kappa_score(Yv, pr_, weights="quadratic")
                    if sc > best: best, bt = sc, t_
                thr[k] = float(bt)
        D["pred_lvl"] = (P >= np.array(thr)).sum(1); out["thr"] = thr
        D["P1"], D["P2"], D["P3"] = P[:, 0], P[:, 1], P[:, 2]
        # trip-bootstrap ensemble (uncertainty)
        rng = np.random.RandomState(seed); ens = []
        for b in range(5):
            ids = list(rng.choice(tr_ids, len(tr_ids), replace=True)); Xb = pd.concat([D[D.trip == i] for i in ids])
            mb = fit_xgb(Xb[F_ORD], (Xb.lvl_fwd >= 1).astype(int).values, D.loc[va, F_ORD], y1[va], seed + b + 1); ens.append(pp(mb, D[F_ORD]))
        ens = np.array(ens); D["p_ens"] = ens.mean(0); D["p_std"] = ens.std(0); D["p_single"] = P[:, 0]
        out["models"] = ords
    if injection:
        cols = F_ORD
        ref = {k: ecdf(D.loc[tr, k].values) for k in ("if1", "if2", "if3", "d_hist", "excess_dwell")}
        D["vz"] = D.v_z_mean5.abs(); ref["vz"] = ecdf(D.loc[tr, "vz"].values)
        if not light: ref["xgb"] = ecdf(D.loc[tr, "P1"].values)
        def detectors(F):
            s = {k: ref[k](F[k].values) for k in ("if1", "if2", "if3", "d_hist", "excess_dwell", "vz")}
            if not light: s["xgb"] = ref["xgb"](pp(out["models"][0], F[cols]))
            else: s["xgb"] = np.zeros(len(F))
            return s
        neg = detectors(D[te]); store = []
        for kind, tid, mask, ws, Fi in build_injected(te_ids, c, seed):
            Fi = apply_ctx(Fi, c)
            Fi["if1"] = -if1.score_samples(Fi[V1]); Fi["if2"] = -if2.score_samples(Fi[V2]); Fi["if3"] = -if3.score_samples(Fi[V3]); Fi["vz"] = Fi.v_z_mean5.abs()
            lvl, _ = levels(Fi, taus); Fi["lvl_now"] = lvl; s = detectors(Fi)
            lead = np.zeros(len(Fi), bool); lead[max(0, ws - H):ws] = True; lead &= ~mask
            store.append((kind, {k: v[mask] for k, v in s.items()}, {k: v[lead] for k, v in s.items()}, lvl[mask], Fi, s, mask))
        out["neg"] = neg; out["inj"] = store; out["neg_lvl"] = D.lvl_now.values[te]
    return out

# ------------------------------------------------------------------ 1) rolling-origin folds (time-aware baseline = main)
FOLDS = [(order[:k - 6], order[k - 6:k], order[k:k + 6]) for k in (24, 30, 36, 42, 48, 54)]
folds_out, oof = [], []
pool_neg = {k: [] for k in DET}; pool_pos = {k: {d: [] for d in DET} for k in KINDS}; pool_lead = {k: {d: [] for d in DET} for k in KINDS}
pool_lvl = {k: [] for k in KINDS}; neg_lvl = []; demo = {}
for fi, (trn, van, ten) in enumerate(FOLDS):
    o = run_fold(trn, van, ten, mode="time", injection=True)
    folds_out.append(dict(fore=o["fore"], base=o["base"], taus=[float(v) for v in o["taus"]], thr=o["thr"]))
    D, te = o["D"], o["te"]; T = D[te].copy(); T["fold"] = fi
    T["order"] = T.trip.map({tid: dep[tid].value for tid in ten}); oof.append(T)
    for d in DET: pool_neg[d].append(o["neg"][d])
    neg_lvl.append(o["neg_lvl"])
    for kind, pos, lead, lvl, Fi, s, mask in o["inj"]:
        for d in DET: pool_pos[kind][d].append(pos[d]); pool_lead[kind][d].append(lead[d])
        pool_lvl[kind].append(lvl)
        if fi == 4 and kind not in demo:
            demo[kind] = dict(t=[int(v) for v in Fi.elapsed], mask=[int(v) for v in mask], speed_kmh=[round(float(v * 3.6), 1) for v in Fi.v],
                              **{k: [round(float(v), 3) for v in s[k]] for k in ("if3", "xgb", "d_hist", "excess_dwell", "vz")})
OOF = pd.concat(oof, ignore_index=True)
neg_all = {d: np.concatenate(pool_neg[d]) for d in DET}
inj_tab = {}
for kind in KINDS:
    inj_tab[kind] = {}
    for d in DET:
        pos = np.concatenate(pool_pos[kind][d]); ld = np.concatenate(pool_lead[kind][d])
        y = np.r_[np.zeros(len(neg_all[d])), np.ones(len(pos))]; sc = np.r_[neg_all[d], pos]
        yl = np.r_[np.zeros(len(neg_all[d])), np.ones(len(ld))]; sl = np.r_[neg_all[d], ld]
        inj_tab[kind][d] = dict(auc=safe_roc(y, sc), tpr5=tpr_at_fpr(neg_all[d], pos), lead_auc=safe_roc(yl, sl) if len(ld) else np.nan, n_pos=int(len(pos)))
lvl_tab = {}
nl = np.concatenate(neg_lvl)
for kind in KINDS:
    lv = np.concatenate(pool_lvl[kind]); lvl_tab[kind] = dict(expected=EXPECT[kind], dist=[float((lv == k).mean()) for k in range(4)], mean=float(lv.mean()),
                                                              at_least_expected=float((lv >= EXPECT[kind]).mean()))
lvl_tab["clean"] = dict(expected=0, dist=[float((nl == k).mean()) for k in range(4)], mean=float(nl.mean()), at_least_expected=1.0)

def fsum(tname, g, key=0):
    v = [f["fore"][tname][g][key] for f in folds_out]; return float(np.nanmean(v)), float(np.nanstd(v))
fore_tab = {t: {g: dict(prauc=fsum(t, g, 0)[0], sd=fsum(t, g, 0)[1], roc=fsum(t, g, 1)[0]) for g in folds_out[0]["fore"][t]} for t in folds_out[0]["fore"]}
base_rates = {t: float(np.mean([f["base"][t] for f in folds_out])) for t in folds_out[0]["base"]}

# ------------------------------------------------------------------ 2) time-aware baseline ablation (injection detection by baseline mode)
base_abl = {}
for mode in ("none", "ampm", "time"):
    if mode == "time": src_inj = inj_tab; res = {k: dict(auc=src_inj[k]["if3"]["auc"], tpr5=src_inj[k]["if3"]["tpr5"], vz_auc=src_inj[k]["vz"]["auc"]) for k in KINDS}
    else:
        pn = []; pp_ = {k: [] for k in KINDS}; pv = {k: [] for k in KINDS}
        for trn, van, ten in FOLDS:
            o = run_fold(trn, van, ten, mode=mode, injection=True, light=True); pn.append(o["neg"])
            for kind, pos, lead, lvl, Fi, s, mask in o["inj"]: pp_[kind].append(pos["if3"]); pv[kind].append(pos["vz"])
        nn = np.concatenate([p["if3"] for p in pn]); nv = np.concatenate([p["vz"] for p in pn]); res = {}
        for kind in KINDS:
            pos = np.concatenate(pp_[kind]); y = np.r_[np.zeros(len(nn)), np.ones(len(pos))]
            pvz = np.concatenate(pv[kind]); yz = np.r_[np.zeros(len(nv)), np.ones(len(pvz))]
            res[kind] = dict(auc=safe_roc(y, np.r_[nn, pos]), tpr5=tpr_at_fpr(nn, pos), vz_auc=safe_roc(yz, np.r_[nv, pvz]))
    base_abl[mode] = res
blocks = pd.concat([FR[i] for i in order]).tb.value_counts().sort_index().to_dict()

# ------------------------------------------------------------------ 3) ordinal evaluation (pooled out-of-fold)
Y = OOF.lvl_fwd.values; P = OOF[["P1", "P2", "P3"]].values
pred = OOF.pred_lvl.values; now = OOF.lvl_now.values; pred05 = (P >= 0.5).sum(1)
def ord_metrics(pr):
    return dict(qwk=float(cohen_kappa_score(Y, pr, weights="quadratic")), mae=float(np.abs(Y - pr).mean()), acc=float((Y == pr).mean()),
                within1=float((np.abs(Y - pr) <= 1).mean()))
ordinal = dict(thresholds=[f["thr"] for f in folds_out], model_default05=ord_metrics(pred05), model=ord_metrics(pred), persistence=ord_metrics(now), all_normal=ord_metrics(np.zeros_like(Y)),
               confusion=[[int(((Y == i) & (pred == j)).sum()) for j in range(4)] for i in range(4)],
               true_dist=[int((Y == k).sum()) for k in range(4)],
               auc_ge1=safe_roc((Y >= 1).astype(int), P[:, 0]), auc_ge2=safe_roc((Y >= 2).astype(int), P[:, 1]),
               auc_ge3=safe_roc((Y >= 3).astype(int), P[:, 2]), n=int(len(Y)))

# ------------------------------------------------------------------ 4) calibration (calibrator fit on EARLIER folds only)
def ece(y, p, bins=8):
    q = np.quantile(p, np.linspace(0, 1, bins + 1)); q[-1] += 1e-9; e = 0
    for lo, hi in zip(q[:-1], q[1:]):
        m = (p >= lo) & (p < hi)
        if m.sum(): e += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(e)
logit = lambda p: np.log(np.clip(p, 1e-4, 1 - 1e-4) / (1 - np.clip(p, 1e-4, 1 - 1e-4))).reshape(-1, 1)
yb = (OOF.lvl_fwd >= 1).astype(int).values; praw = OOF.p_ens.values
cal = {"Raw ensemble mean": np.zeros(len(OOF)), "Platt scaling": np.zeros(len(OOF)), "Isotonic": np.zeros(len(OOF))}
use = OOF.fold.values >= 1
for f in range(1, len(FOLDS)):
    prev = OOF.fold.values < f; cur = OOF.fold.values == f
    cal["Raw ensemble mean"][cur] = praw[cur]
    cal["Platt scaling"][cur] = LogisticRegression(C=1e3).fit(logit(praw[prev]), yb[prev]).predict_proba(logit(praw[cur]))[:, 1]
    cal["Isotonic"][cur] = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(praw[prev], yb[prev]).predict(praw[cur])
cal_tab = {k: dict(brier=float(brier_score_loss(yb[use], np.clip(v[use], 0, 1))), ece=ece(yb[use], np.clip(v[use], 0, 1)),
                   logloss=float(log_loss(yb[use], np.clip(v[use], 1e-4, 1 - 1e-4)))) for k, v in cal.items()}
def rel(p, y, bins=6):
    q = np.quantile(p, np.linspace(0, 1, bins + 1)); q[-1] += 1e-9; r = []
    for lo, hi in zip(q[:-1], q[1:]):
        m = (p >= lo) & (p < hi)
        if m.sum(): r.append(dict(pred=round(float(p[m].mean()), 3), obs=round(float(y[m].mean()), 3), n=int(m.sum())))
    return r
reliab = {k: rel(np.clip(v[use], 0, 1), yb[use]) for k, v in cal.items()}
best_cal = min(cal_tab, key=lambda k: cal_tab[k]["brier"])

# ------------------------------------------------------------------ 5) uncertainty
# 5a) ensemble spread vs error
pst = OOF.p_std.values; err = np.abs(yb - praw); terc = np.quantile(pst, [1 / 3, 2 / 3])
ens_tab = []
for name, m in (("low spread", pst <= terc[0]), ("medium", (pst > terc[0]) & (pst <= terc[1])), ("high spread", pst > terc[1])):
    ens_tab.append(dict(group=name, n=int(m.sum()), mean_spread=float(pst[m].mean()), brier=float(((praw[m] - yb[m]) ** 2).mean()),
                        mean_abs_err=float(err[m].mean())))
# 5b) conformal prediction sets (binary labels), calibration from earlier folds, plus adaptive version
# FIX (v3.1): fold-0 rows have no calibrator; use their raw probabilities, not zeros, when they serve as conformal calibration data
cal['Platt scaling'][OOF.fold.values == 0] = praw[OOF.fold.values == 0]
ALPHA = 0.10; GAMMA = 0.02
conf = []
pc_all = cal["Platt scaling"]
sets_out = np.full(len(OOF), -1)
for f in range(1, len(FOLDS)):
    prev = OOF.fold.values < f; cur = np.where(OOF.fold.values == f)[0]
    pcal = np.clip(cal["Platt scaling"][prev], 1e-4, 1 - 1e-4); ycal = yb[prev]
    sc_cal = np.where(ycal == 1, 1 - pcal, pcal)                        # nonconformity = 1 - p(true class)
    n = len(sc_cal); q = np.quantile(sc_cal, min(1.0, math.ceil((n + 1) * (1 - ALPHA)) / n))
    p = np.clip(cal["Platt scaling"][cur], 1e-4, 1 - 1e-4); y = yb[cur]
    in1 = (1 - p) <= q; in0 = p <= q                                    # class 1 included if 1-p<=q ; class 0 if p<=q
    cover = np.where(y == 1, in1, in0); size = in0.astype(int) + in1.astype(int)
    # adaptive conformal inference over the time-ordered stream
    order_idx = np.argsort(OOF["order"].values[cur] * 1e3 + OOF.elapsed.values[cur], kind="stable"); a_t = ALPHA; ac = []; asz = []
    for j in order_idx:
        qq = np.quantile(sc_cal, min(1.0, max(0.0, 1 - a_t)))
        i1, i0 = (1 - p[j]) <= qq, p[j] <= qq; cv = bool(i1) if y[j] == 1 else bool(i0); ac.append(cv); asz.append(int(i1) + int(i0))
        a_t = float(np.clip(a_t + GAMMA * (ALPHA - (0 if cv else 1)), 0.001, 0.999))
    sing = size == 1
    conf.append(dict(fold=f, n=int(len(cur)), static_cov=float(cover.mean()), static_size=float(size.mean()), uncertain=float((size == 2).mean()),
                     empty=float((size == 0).mean()), singleton_acc=float(cover[sing].mean()) if sing.any() else np.nan,
                     adaptive_cov=float(np.mean(ac)), adaptive_size=float(np.mean(asz))))
    sets_out[cur] = size
conf_sum = dict(target=1 - ALPHA, static_cov=float(np.mean([c_["static_cov"] for c_ in conf])), adaptive_cov=float(np.mean([c_["adaptive_cov"] for c_ in conf])),
                static_uncertain=float(np.mean([c_["uncertain"] for c_ in conf])), singleton_acc=float(np.nanmean([c_["singleton_acc"] for c_ in conf])),
                static_cov_min=float(np.min([c_["static_cov"] for c_ in conf])), adaptive_uncertain=float(np.mean([c_["adaptive_size"] > 1 for c_ in conf])))

conf_sweep = []
for al in (0.1, 0.2, 0.3, 0.4):
    cv_, sg_, sa_, sz_ = [], [], [], []
    for f in range(1, len(FOLDS)):
        prev = OOF.fold.values < f; cur = OOF.fold.values == f
        pcal = np.clip(cal["Platt scaling"][prev], 1e-4, 1 - 1e-4); sc_ = np.where(yb[prev] == 1, 1 - pcal, pcal); n_ = len(sc_)
        q_ = np.quantile(sc_, min(1.0, math.ceil((n_ + 1) * (1 - al)) / n_)); p_ = np.clip(cal["Platt scaling"][cur], 1e-4, 1 - 1e-4); y_ = yb[cur]
        a1, a0 = (1 - p_) <= q_, p_ <= q_; cov_ = np.where(y_ == 1, a1, a0); sz = a1.astype(int) + a0.astype(int); sing = sz == 1
        cv_.append(cov_.mean()); sg_.append(sing.mean()); sa_.append(cov_[sing].mean() if sing.any() else np.nan); sz_.append(sz.mean())
    conf_sweep.append(dict(alpha=al, target=1 - al, coverage=float(np.mean(cv_)), singleton_share=float(np.mean(sg_)), singleton_acc=float(np.nanmean(sa_)), size=float(np.mean(sz_))))

# ------------------------------------------------------------------ 6) facts + sample export
call = fit_ctx(order, "time"); Dall = apply_ctx(pd.concat([FR[i] for i in order], ignore_index=True), call)
facts = dict(blocks={str(k): int(v) for k, v in blocks.items()}, n=int(len(Dall)))

# sample: last fold (k=54), trip with most concern; calibrator = Platt fit on all earlier folds
last = OOF[OOF.fold == len(FOLDS) - 1].copy(); prevm = OOF.fold.values < len(FOLDS) - 1
plat = LogisticRegression(C=1e3).fit(logit(praw[prevm]), yb[prevm])
pcal_q = np.clip(cal["Platt scaling"][prevm], 1e-4, 1 - 1e-4); sc_cal = np.where(yb[prevm] == 1, 1 - pcal_q, pcal_q)
ALPHA_S = 0.10   # 90% target; empirical coverage is checked in conf_sweep
qf = np.quantile(sc_cal, min(1.0, math.ceil((len(sc_cal) + 1) * (1 - ALPHA_S)) / len(sc_cal)))
last["risk"] = plat.predict_proba(logit(last.p_ens.values))[:, 1]
last["lo"] = np.clip(last.p_ens - last.p_std, 0, 1); last["hi"] = np.clip(last.p_ens + last.p_std, 0, 1)
in1 = (1 - last.risk) <= qf; in0 = last.risk <= qf
last["pred_set"] = np.where(in1 & in0, "uncertain", np.where(in1, "concern likely", "normal likely"))
last["next_level"] = last.pred_lvl.values
pick = last.groupby("trip").lvl_fwd.sum().idxmax(); s = last[last.trip == pick].sort_values("elapsed")
smp = pd.DataFrame(dict(
    time_local=s.t.dt.tz_convert("Asia/Manila").dt.strftime("%Y-%m-%d %H:%M:%S"), latitude=(LAT0 + s.y / KY).round(6), longitude=(LON0 + s.x / KX).round(6),
    speed_kmh=(s.v * 3.6).round(1), concern_now=s.lvl_now.map(dict(enumerate(LEVELS))), risk_next_60s=s.risk.round(3), risk_low=s.lo.round(3), risk_high=s.hi.round(3),
    safety_index=(100 * (1 - s.risk)).round(1), p_minor_or_worse=s.P1.round(3), p_major_or_worse=s.P2.round(3), p_critical=s.P3.round(3),
    predicted_next_level=s.next_level.map(dict(enumerate(LEVELS))), confidence_set=s.pred_set,
    actual_worst_level_next_60s=s.lvl_fwd.map(dict(enumerate(LEVELS)))))
smp.to_csv(SAMPLE, index=False)

res = dict(fore=fore_tab, base_rates=base_rates, inj=inj_tab, lvl_tab=lvl_tab, base_abl=base_abl, blocks=facts["blocks"], ordinal=ordinal,
           cal=cal_tab, reliab=reliab, best_cal=best_cal, ens=ens_tab, conf=conf, conf_sum=conf_sum, conf_sweep=conf_sweep, demo=demo,
           params=dict(H=H, bands=BANDS, dwell_levels=DWELL_LV, if_q=IF_Q, shrink_k=SHRINK_K, kinds=KINDS, alpha=ALPHA, alpha_sample=0.10, gamma=GAMMA,
                       n_oof=int(len(OOF)), n_oof_cal=int(use.sum()), sample_trip=int(pick), sample_rows=int(len(smp))),
           n_neg=int(len(neg_all["if3"])))
json.dump(res, open(OUT, "w"), indent=1, default=float)
print(json.dumps(dict(fore=fore_tab, base_rates=base_rates, ordinal=ordinal, cal=cal_tab, ens=ens_tab, conf_sum=conf_sum, blocks=facts["blocks"], best_cal=best_cal), indent=1, default=float))
print("INJ")
for k in KINDS:
    print(k, {d: (round(v["auc"], 2), round(v["tpr5"], 2), None if v["lead_auc"] != v["lead_auc"] else round(v["lead_auc"], 2)) for d, v in inj_tab[k].items()})
print("BASE ABL")
for m, r in base_abl.items(): print(m, {k: (round(v["auc"], 2), round(v["vz_auc"], 2)) for k, v in r.items()})
print("LEVELS", {k: [round(x, 2) for x in v["dist"]] for k, v in lvl_tab.items()})
