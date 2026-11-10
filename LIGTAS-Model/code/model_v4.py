"""GPS movement scoring and short-horizon concern forecasts for one commuter."""
import json, math, os, glob
import numpy as np, pandas as pd
from scipy.spatial import cKDTree
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression
import xgboost as xgb

SEED = 7
MAX_SPEED, MAX_ACC, MAX_ERR = 50.0, 10.0, 50.0      # Verma et al. (2024): m/s, m/s^2, m
HORIZON_S = 60.0
SEG = 100.0
BANDS = [150.0, 500.0, 1000.0]                      # Edwards & Hosein (2025)
ZCUT = [2.5, 3.0, 3.5]                              # robust-z cut-offs: Leys et al. (2013) 2.5; Iglewicz & Hoaglin 3.5 (NIST); middle level halfway
ROUTE_MODE = "history"                              # use the planned route and variants supported by earlier trips
ROUTE_MIN_TRIPS = 5                                  # earlier trips in the same direction needed before distance from earlier trips is used
OFFROUTE_LEVEL = False                               # extra concern level from how long the vehicle stays off the route, judged against this commuter's own earlier off-route episodes
STOP_V = 5.0 / 3.6                                   # 5 km/h in m/s: low-speed limit used by Sabir et al. (2025)
SHRINK_K = 8.0
LEVELS = ["Normal", "Minor", "Major", "Critical"]
LAT0, LON0 = 18.056, 120.556
KX, KY = 111320 * np.cos(np.radians(LAT0)), 110574.0
DT = 10.0                                            # nominal sampling interval, for reporting
MAX_GAP_S = 25.0                                     # elapsed time above this breaks continuity
ROUTE_SUPPORT = 2                                    # distinct trips needed to support a route variant

def to_xy(lat, lon): return (np.asarray(lon, float) - LON0) * KX, (np.asarray(lat, float) - LAT0) * KY
wrap = lambda a: (a + np.pi) % (2 * np.pi) - np.pi
def mad(a): a = np.asarray(a); return float(np.median(np.abs(a - np.median(a))) * 1.4826)
TOD_MODE = "four"                                    # "four" parts of the day, "two" (night / day), or "none" (one all-day baseline)
def hour_block(h):
    if TOD_MODE == "none": return np.zeros(np.shape(h), int)
    if TOD_MODE == "two": return np.where((h < 6) | (h >= 19), 0, 1)
    return np.where((h < 6) | (h >= 19), 0, np.where(h < 10, 1, np.where(h < 14, 2, 3)))

# ---------------------------------------------------------------- route reference
def polyline(coords):
    c = np.array(coords); x, y = to_xy(c[:, 1], c[:, 0]); P = np.c_[x, y]
    return P, np.r_[0, np.cumsum(np.hypot(*np.diff(P, axis=0).T))]

def project(P, cum, px, py):
    A, B = P[:-1], P[1:]; AB = B - A; L2 = (AB ** 2).sum(1) + 1e-9
    s = np.zeros(len(px)); d = np.zeros(len(px)); brg = np.zeros(len(px))
    for i, (x, y) in enumerate(zip(px, py)):
        t = np.clip(((x - A[:, 0]) * AB[:, 0] + (y - A[:, 1]) * AB[:, 1]) / L2, 0, 1)
        dd = np.hypot(x - (A[:, 0] + t * AB[:, 0]), y - (A[:, 1] + t * AB[:, 1])); k = dd.argmin()
        s[i] = cum[k] + t[k] * np.sqrt(L2[k]); d[i] = dd[k]; brg[i] = math.atan2(AB[k, 1], AB[k, 0])
    return s, d, brg

# ---------------------------------------------------------------- 1) cleaning
def clean_trip(df):
    """Return (clean DataFrame with columns t, x, y) and a dict of what was removed."""
    rep = dict(raw=len(df))
    t = pd.to_datetime(df["timestamp"], format="mixed", utc=True, errors="coerce")
    lat = pd.to_numeric(df.latitude, errors="coerce")
    lon = pd.to_numeric(df.longitude, errors="coerce")
    d = df.assign(t=t, latitude=lat, longitude=lon)
    ok = t.notna() & lat.between(-90, 90) & lon.between(-180, 180)
    rep["bad_rows"] = int((~ok).sum())
    d = d[ok].sort_values("t", kind="stable")
    rep["duplicates"] = int(d.t.duplicated().sum())
    d = d.drop_duplicates("t")
    rep["poor_accuracy"] = 0
    if "accuracy" in d.columns:
        accuracy = pd.to_numeric(d.accuracy, errors="coerce")
        bad = accuracy.notna() & ~accuracy.between(0, MAX_ERR)
        rep["poor_accuracy"] = int(bad.sum())
        d = d[~bad]
    x, y = to_xy(d.latitude, d.longitude)
    tt = d.t.diff().dt.total_seconds().fillna(0).cumsum().to_numpy()
    keep = np.ones(len(d), bool)
    last, last_speed = 0, None
    n_speed = n_acc = 0
    for i in range(1, len(d)):
        dt = tt[i] - tt[last]
        if dt > MAX_GAP_S:
            last, last_speed = i, None
            continue
        speed = math.hypot(x[i] - x[last], y[i] - y[last]) / dt
        if speed > MAX_SPEED:
            keep[i] = False
            n_speed += 1
            continue
        if last_speed is not None and abs(speed - last_speed) / dt > MAX_ACC:
            keep[i] = False
            n_acc += 1
            continue
        last, last_speed = i, speed
    rep["speed_over_50"] = n_speed; rep["accel_over_10"] = n_acc
    out = pd.DataFrame(dict(t=pd.DatetimeIndex(d.t[keep]), x=x[keep], y=y[keep]))
    rep["kept"] = len(out); rep["removed"] = rep["raw"] - rep["kept"]
    rep["usable"] = bool(len(out) >= 20 and rep["removed"] <= 0.2 * max(rep["raw"], 1)
                         and out.t.diff().dt.total_seconds().le(MAX_GAP_S).any())
    return out, rep

# ---------------------------------------------------------------- 2) per-trip features (no training needed)
def rolling(F, column, seconds, method, min_periods=1):
    """Time-based rolling statistic, restarted at each break in GPS coverage."""
    out = np.zeros(len(F))
    for _, g in F.groupby(["trip", "segment"], sort=False):
        values = pd.Series(g[column].to_numpy(), index=pd.DatetimeIndex(g.t))
        window = values.rolling(pd.Timedelta(seconds=seconds), min_periods=min_periods)
        out[g.index] = getattr(window, method)()
    return out

def make_frame(tid, t, x, y, key, PATH):
    elapsed = (t - t[0]).total_seconds().to_numpy()
    n = len(t)
    dt = np.diff(elapsed, prepend=0)
    observed = (dt > 0) & (dt <= MAX_GAP_S)
    segment = np.cumsum(~observed)
    start = np.maximum.accumulate(np.where(observed, 0, np.arange(n)))
    prev = np.maximum(np.searchsorted(elapsed, elapsed - 3, side="right") - 1, start)
    dx, dy = x - x[prev], y - y[prev]
    interval = elapsed - elapsed[prev]
    v = np.divide(np.hypot(dx, dy), interval, out=np.zeros(n), where=interval > 0)
    paired = observed & np.r_[False, observed[:-1]]
    a = np.divide(np.diff(v, prepend=0), dt, out=np.zeros(n), where=paired)
    jerk = np.divide(np.diff(a, prepend=0), dt, out=np.zeros(n), where=paired & np.r_[False, paired[:-1]])
    hd = np.arctan2(dy, dx)
    moving = v >= STOP_V
    turn = np.divide(np.abs(wrap(np.diff(hd, prepend=0))), dt, out=np.zeros(n),
                     where=paired & moving & np.r_[False, moving[:-1]])
    P, cum = PATH[key]
    s, d, brg = project(P, cum, x, y)
    hdev = np.where(moving, np.degrees(np.abs(wrap(hd - brg))), 0)
    loc = t.tz_convert("Asia/Manila")
    f = pd.DataFrame(dict(trip=tid, dir=int(key == "return"), t=t, x=x, y=y, dt=dt, v=v, a=a,
                          jerk=jerk, turn=turn, s=s, d=d, heading_dev=hdev, hour=loc.hour + loc.minute / 60,
                          dow=loc.dayofweek, elapsed=elapsed, observed=observed, segment=segment))
    f["tb"] = hour_block(f.hour.values)
    f["stopped"] = (observed & ~moving).astype(int)
    f["abs_a"] = f.a.abs()
    for name, column, method in [
        ("v_mean6", "v", "mean"), ("v_std6", "v", "std"), ("a_min6", "a", "min"),
        ("a_absmax6", "abs_a", "max"), ("stops6", "stopped", "sum"), ("turn_sum6", "turn", "sum"),
    ]:
        f[name] = rolling(f, column, 60, method)
    f["v_std6"] = f.v_std6.fillna(0)
    run, tss = np.zeros(n), np.zeros(n)
    dwell, since = 0.0, 600.0
    for i, stopped in enumerate(f.stopped):
        dwell = dwell + dt[i] if stopped else 0.0
        since = 600.0 if not observed[i] else (0.0 if stopped else min(since + dt[i], 600.0))
        run[i], tss[i] = dwell, since
    f["dwell"], f["time_since_stop"] = run, tss
    f["cum_stop"] = (f.stopped * f.dt).cumsum()
    f["sbin"] = (f.s // SEG).astype(int).clip(lower=0)
    prev30 = np.searchsorted(elapsed, elapsed - 30, side="right") - 1
    valid = prev30 >= start
    prev30 = np.maximum(prev30, 0)
    delta = elapsed - elapsed[prev30]
    f["v_trend3"] = np.divide(v - v[prev30], delta, out=np.zeros(n), where=valid)
    f["d_rate3"] = np.divide(d - d[prev30], delta, out=np.zeros(n), where=valid)
    f["a_mean3"] = rolling(f, "a", 30, "mean")
    prev10 = np.searchsorted(elapsed, elapsed - 10, side="right") - 1
    f["hdev_rate"] = np.where(prev10 >= start, hdev - hdev[np.maximum(prev10, 0)], 0)
    f["v_run"] = f.v.where(f.v > STOP_V).expanding().mean().fillna(0)
    f["v_rel"] = f.v / (f.v_run + 1.0)
    return f

# ---------------------------------------------------------------- 3) dataset loading
class Data: pass

def load_dataset(ds, verbose=True):
    """Reads <ds>/project.json (route reference) and every CSV in <ds>/a-to-b and <ds>/b-to-a."""
    global DT
    proj = json.load(open(os.path.join(ds, "project.json")))["routes"][0]
    PATH = {"outbound": polyline(proj["paths"]["outbound"]["coords"]), "return": polyline(proj["paths"]["return"]["coords"])}
    items = []
    for folder, key in (("a-to-b", "outbound"), ("b-to-a", "return")):
        for fp in sorted(glob.glob(os.path.join(ds, folder, "*.csv"))):
            items.append((os.path.relpath(fp, ds), key, pd.read_csv(fp)))
    cleaned, reports, steps = [], {}, []
    for fn, key, df in items:
        c, rep = clean_trip(df); reports[fn] = rep
        if rep["usable"] and len(c) > 1: cleaned.append((fn, key, c)); steps.append(c.t.diff().dt.total_seconds().median())
    DT = float(np.median(steps)) if steps else None
    D = Data(); D.files, D.key, D.FR, D.dep, D.clean_report, D.dt = [], [], {}, [], reports, DT
    D.raw_clean = {}
    for i, (fn, key, c) in enumerate(cleaned):
        D.files.append(fn); D.key.append(key); D.dep.append(c.t.iloc[0]); D.raw_clean[i] = c
        D.FR[i] = make_frame(i, pd.DatetimeIndex(c.t), c.x.values, c.y.values, key, PATH)
    D.end = {i: c.t.iloc[-1] for i, c in D.raw_clean.items()}
    D.dep = pd.Series(D.dep, dtype="datetime64[ns, UTC]"); D.order = list(D.dep.sort_values().index); D.PATH = PATH
    if verbose: print(f"loaded {len(D.files)} usable trips of {len(items)}; recording interval {DT} s")
    return D

# ---------------------------------------------------------------- 4) context learned from the history that is passed in
class Ctx: pass

def fit_ctx(FR, train_ids):
    Ft = pd.concat([FR[i] for i in train_ids], ignore_index=True); c = Ctx()
    mv = Ft[Ft.v > STOP_V]
    c.gm, c.gs = (0.0, 1.0) if mv.empty else (float(mv.v.median()), max(mad(mv.v), 1.0))
    def table(keys):
        groups = mv.groupby(keys)
        return {key: (float(g.v.median()), max(mad(g.v), 1.0), g.trip.nunique()) for key, g in groups}
    c.seg = table(["sbin", "dir"]); c.tb = table(["sbin", "dir", "tb"])
    # A stop ending at signal loss or recording end is censored.
    next_moving = Ft.observed.shift(-1, fill_value=False) & ~Ft.stopped.shift(-1, fill_value=0).astype(bool)
    completed = Ft[(Ft.dwell > 0) & next_moving & Ft.trip.eq(Ft.trip.shift(-1))]
    def stop_table(keys):
        refs = {}
        for key, g in completed.groupby(keys):
            if g.trip.nunique() >= 3:
                refs[key] = (g.dwell.median(), max(mad(g.dwell), 10.0))
        return refs
    c.stop_local = stop_table(["sbin", "dir"])
    c.stop_dir = stop_table("dir")
    c.stop_global = (completed.dwell.median(), max(mad(completed.dwell), 10.0)) if completed.trip.nunique() >= 3 else None
    c.pstop = Ft.groupby(["sbin", "dir"]).stopped.mean().to_dict()
    c.zone = {dr: np.array(sorted(b for (b, d_), p in c.pstop.items() if d_ == dr and p >= 0.25)) for dr in (0, 1)}
    c.trees = {0: {}, 1: {}}
    for i in train_ids:
        f = FR[i]; c.trees[int(f.dir.iloc[0])][i] = cKDTree(np.c_[f.x, f.y])
    return c

def baseline(F, c):
    est = np.full(len(F), c.gm); sig = np.full(len(F), c.gs)
    for i, (sb, dr, tb) in enumerate(zip(F.sbin.values, F.dir.values, F.tb.values)):
        m0 = c.seg.get((sb, dr)); l1 = c.tb.get((sb, dr, tb))
        if m0 is None: continue
        if l1 is None: est[i], sig[i] = m0[0], m0[1]
        else:
            w = l1[2] / (l1[2] + SHRINK_K); est[i] = w * l1[0] + (1 - w) * m0[0]; sig[i] = w * l1[1] + (1 - w) * m0[1]
    return est, sig

def apply_ctx(F, c):
    F = F.reset_index(drop=True).copy()
    est, sig = baseline(F, c); F["v_resid"] = F.v - est; F["v_z"] = np.clip((F.v - est) / sig, -6, 6)
    F["v_z_mean5"] = rolling(F, "v_z", 50, "mean")
    refs = [c.stop_local.get((sb, dr), c.stop_dir.get(dr, c.stop_global)) for sb, dr in zip(F.sbin, F.dir)]
    ref = np.array([r if r is not None else (0.0, np.inf) for r in refs])
    F["excess_dwell"] = (F.dwell - ref[:, 0]).clip(lower=0)
    F["stop_z"] = F.excess_dwell / ref[:, 1]
    F["stop_unexp"] = F.stopped * (1 - np.array([c.pstop.get(k, 0.0) for k in zip(F.sbin, F.dir)]))
    dh = F.d.to_numpy().copy(); n_other = np.zeros(len(F), int)
    for tid, g in F.groupby("trip"):
        trees = [tr for j, tr in c.trees[int(g.dir.iloc[0])].items() if j != tid]
        n_other[g.index] = len(trees)
        if len(trees) >= ROUTE_SUPPORT:
            distances = np.array([tr.query(np.c_[g.x, g.y])[0] for tr in trees])
            dh[g.index] = np.partition(distances, ROUTE_SUPPORT - 1, axis=0)[ROUTE_SUPPORT - 1]
    F["d_hist"] = dh; F["n_same_dir"] = n_other
    F["band_hist"] = np.digitize(F.d_hist, BANDS, right=True); F["band_poly"] = np.digitize(F.d, BANDS, right=True)
    enough = F.n_same_dir >= max(ROUTE_MIN_TRIPS, ROUTE_SUPPORT)
    combined = np.maximum(F.band_hist, F.band_poly) if ROUTE_MODE == "max" else np.minimum(F.band_hist, F.band_poly)
    F["band_route"] = np.where(enough, combined, F.band_poly)
    dr_ = np.zeros(len(F))
    for tid, g in F.groupby("trip"):
        run = 0.0
        for i, b, dt, observed in zip(g.index, g.band_route, g.dt, g.observed):
            run = run + dt if observed and b >= 1 else 0.0
            dr_[i] = run
    F["dev_run"] = dr_
    dz = np.full(len(F), 600.0)
    for i, (s, dr) in enumerate(zip(F.s.values, F.dir.values)):
        z = c.zone[int(dr)]
        if len(z):
            delta = (z * SEG + SEG / 2) - s; ah = delta[delta >= -20]
            if len(ah): dz[i] = min(ah.min(), 600.0)
    F["dist_to_stop_zone"] = np.maximum(dz, 0)
    return F

V3 = ["v", "a", "jerk", "turn", "d", "v_resid", "v_std6", "dwell", "heading_dev", "excess_dwell", "d_hist", "v_z", "v_z_mean5", "v_trend3", "d_rate3", "time_since_stop"]
BASE = ["v", "a", "jerk", "turn", "d", "v_mean6", "v_std6", "a_min6", "a_absmax6", "stops6", "dwell", "turn_sum6", "s", "dir", "hour", "dow", "elapsed", "v_resid"]
DEVG = ["d_hist", "band_hist", "band_poly", "band_route", "heading_dev", "dev_run"]
DWG = ["excess_dwell", "stop_unexp", "cum_stop"]
RUN = ["v_trend3", "a_mean3", "time_since_stop", "dist_to_stop_zone", "d_rate3", "hdev_rate", "v_z", "v_z_mean5", "v_rel"]
F_ORD = BASE + DEVG + DWG + RUN + ["lvl_now", "if3"]
# ---- settings of the forecaster (defaults reproduce the main results; changed only by the sensitivity checks)
XGB_KW = dict(learning_rate=0.04, max_depth=3, subsample=0.8, colsample_bytree=0.8, min_child_weight=3, reg_lambda=2.0)
DROP_FEATURES = []          # inputs left out of the forecaster
USE_PLATT = True            # calibrate the chances with Platt scaling
CAL_SHARE = 0.2             # latest share of the history trips held out
POS_WEIGHT = False          # give rare unusual moments more weight
MODEL_KIND = "xgb"          # "xgb", or "logistic" (a plain logistic regression, for comparison)
LOGIT_FEATS = ["lvl_now", "if3", "dwell", "dev_run", "d", "v", "v_trend3", "time_since_stop", "stops6", "v_z", "excess_dwell"]

def rz(x, med, sc): return (np.asarray(x, float) - med) / sc

def offroute_durations(FH):
    """lengths (s) of finished off-route episodes in the history (dev_run = seconds continuously beyond the first distance band)"""
    out = []
    for tid, g in FH.groupby("trip"):
        a = g.dev_run.values
        for k in range(len(a)):
            if a[k] > 0 and (k + 1 == len(a) or a[k + 1] == 0): out.append(a[k])
    return np.array(out)

def level_parts(F, m):
    z_if = rz(F.if3.values, m.if_med, m.if_sc); ifl = np.digitize(z_if, ZCUT)
    bl = F.band_route.values
    dl = np.digitize(F.stop_z, ZCUT)
    dl = np.where(F.stopped.values == 1, dl, 0)
    parts = [ifl, bl, dl]
    if OFFROUTE_LEVEL and getattr(m, "off_sc", None) is not None:
        ol = np.digitize(rz(F.dev_run.values, m.off_med, m.off_sc), ZCUT); parts.append(np.where(F.dev_run.values > 0, ol, 0))
    return parts

def levels(F, m): return np.maximum.reduce(level_parts(F, m))

def fwd_max(F, col, horizon):
    """Worst level in (t, t + horizon], or NaN if the window lacks coverage."""
    out = np.full(len(F), np.nan)
    for _, g in F.groupby(["trip", "segment"], sort=False):
        ts, levels = g.elapsed.to_numpy(), g[col].to_numpy()
        ends = np.searchsorted(ts, ts + horizon, side="right")
        for i, end in enumerate(ends):
            if g.observed.iloc[i] and ts[-1] >= ts[i] + horizon and end > i + 1:
                out[g.index[i]] = levels[i + 1:end].max()
    return out

# ---------------------------------------------------------------- 5) fit on a history, then predict a new trip
class Model: pass

def _if_scores(models, X):
    return [-m.score_samples(X) for m in models]

def fit_model(D, hist_ids, seed=SEED):
    """hist_ids: the earlier trips this model is allowed to learn from (the caller applies any expiry)."""
    m = Model()
    ids = sorted(hist_ids, key=lambda i: D.dep[i])
    ncal = min(max(2, round(CAL_SHARE * len(ids))), len(ids) - 2)
    fit_ids, cal_ids = ids[:-ncal], ids[-ncal:]
    m.ctx = fit_ctx(D.FR, fit_ids)
    FH = apply_ctx(pd.concat([D.FR[i] for i in fit_ids], ignore_index=True), m.ctx)
    A, B = set(fit_ids[::2]), set(fit_ids[1::2])

    def fit_forest(trips):
        mask = FH.trip.isin(trips) & FH.observed
        return IsolationForest(n_estimators=100, random_state=seed, n_jobs=2).fit(FH.loc[mask, V3])

    m.ifs = (fit_forest(A), fit_forest(B))
    sA, sB = _if_scores(m.ifs, FH[V3])
    FH["if_raw"] = np.where(FH.trip.isin(A), sB, sA)
    FH["if3"] = rolling(FH, "if_raw", 30, "mean")
    scores = FH.loc[FH.observed, "if3"]
    m.if_med, m.if_sc = float(scores.median()), max(mad(scores), 1e-6)
    od = offroute_durations(FH)
    m.off_med, m.off_sc = (float(np.median(od)), max(mad(od), 10.0)) if len(od) >= 5 else (0.0, None)
    m.h = HORIZON_S
    FH["lvl_now"] = levels(FH, m)
    FH["lvl_fwd"] = fwd_max(FH, "lvl_now", m.h)
    tr = FH.dropna(subset=["lvl_fwd"])
    ca = score_frame(pd.concat([D.FR[i] for i in cal_ids], ignore_index=True), m).dropna(subset=["lvl_fwd"])
    platt_ids, set_ids = cal_ids[:-1], cal_ids[-1:]
    platt = ca[ca.trip.isin(platt_ids)]
    set_cal = ca[ca.trip.isin(set_ids)]
    m.feats = [f for f in (F_ORD if MODEL_KIND == "xgb" else LOGIT_FEATS) if f not in DROP_FEATURES]
    m.models, m.platt, m.prior = {}, {}, {}
    for k in (1, 2, 3):
        ytr, yca = (tr.lvl_fwd >= k).to_numpy(), (platt.lvl_fwd >= k).to_numpy()
        m.prior[k] = float(ytr.mean()) if len(ytr) else 0.0
        mod = None
        if ytr.sum() >= 3 and (~ytr).sum() >= 3:
            if MODEL_KIND == "logistic":
                from sklearn.pipeline import make_pipeline
                from sklearn.preprocessing import StandardScaler
                mod = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=500,
                    class_weight="balanced" if POS_WEIGHT else None)).fit(tr[m.feats], ytr)
            else:
                kw = dict(XGB_KW, random_state=seed, n_jobs=2)
                if POS_WEIGHT:
                    kw["scale_pos_weight"] = float(min(10.0, (~ytr).sum() / ytr.sum()))
                kw.setdefault("n_estimators", 120)
                mod = xgb.XGBClassifier(**kw).fit(tr[m.feats], ytr)
        m.models[k], m.platt[k] = mod, None
        if USE_PLATT and mod is not None and yca.sum() >= 3 and (~yca).sum() >= 3:
            logits = probability_logits(mod.predict_proba(platt[m.feats])[:, 1])
            m.platt[k] = LogisticRegression(C=1e4).fit(logits, yca)
    # Empirical coverage only. Windows from the same trip are dependent.
    n = len(set_cal)
    m.conf_thr = None
    if n >= 20:
        q = _class_probs(_cum_probs(m, set_cal))
        errors = 1 - q[np.arange(n), set_cal.lvl_fwd.to_numpy(dtype=int)]
        rank = math.ceil((n + 1) * 0.9)
        m.conf_thr = float(np.partition(errors, rank - 1)[rank - 1]) if rank <= n else 1.0
    m.n_hist, m.n_cal = len(hist_ids), ncal
    return m

def probability_logits(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p)).reshape(-1, 1)

def _cum_probs(m, F):
    P = np.zeros((len(F), 3))
    for k in (1, 2, 3):
        mod = m.models[k]
        if mod is None:
            P[:, k - 1] = m.prior[k]
            continue
        p = mod.predict_proba(F[m.feats])[:, 1]
        if m.platt[k] is not None:
            p = m.platt[k].predict_proba(probability_logits(p))[:, 1]
        P[:, k - 1] = p
    return np.minimum.accumulate(P, axis=1)                  # P(>=1) >= P(>=2) >= P(>=3)

def _class_probs(P):
    c = np.c_[np.ones(len(P)), P, np.zeros(len(P))]
    return np.clip(c[:, :-1] - c[:, 1:], 0, 1)

def score_frame(frame, m):
    F = apply_ctx(frame, m.ctx)
    sA, sB = _if_scores(m.ifs, F[V3])
    F["if_raw"] = (sA + sB) / 2
    F["if3"] = rolling(F, "if_raw", 30, "mean")
    F["lvl_now"] = levels(F, m)
    F["lvl_fwd"] = fwd_max(F, "lvl_now", m.h)
    return F

def predict_trip(D, m, i):
    F = score_frame(D.FR[i], m)
    P = _cum_probs(m, F)
    q = _class_probs(P)
    parts = level_parts(F, m)
    out = F[["trip", "t", "elapsed", "dt", "observed", "segment", "lvl_now", "lvl_fwd",
             "if3", "d", "d_hist", "excess_dwell"]].copy()
    out["lvl_if"], out["lvl_route"], out["lvl_stop"] = parts[:3]
    out[["P1", "P2", "P3"]] = P
    out["exp_level"] = P.sum(1)
    out["pred_level"] = q.argmax(axis=1)
    out.loc[~out.observed, ["lvl_now", "lvl_if", "lvl_route", "lvl_stop", "P1", "P2", "P3",
                            "exp_level", "pred_level"]] = np.nan
    out["conf_set"] = [
        np.flatnonzero(1 - qq <= m.conf_thr).tolist() if m.conf_thr is not None and observed else None
        for qq, observed in zip(q, out.observed)
    ]
    out["n_hist"] = m.n_hist
    return out
