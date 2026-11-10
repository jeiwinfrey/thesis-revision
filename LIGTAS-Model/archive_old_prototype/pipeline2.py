"""
LIGTAS-CASCADE v2
Adds: route-deviation bands (Edwards & Hosein 2025 scheme), heading deviation, sustained off-route time,
excess/unexpected dwell, a rule-based label channel, a persistence baseline, a rotated (EIF-like) forest,
an injected-anomaly benchmark with exact ground truth, threshold/seed sensitivity and a cold-start curve.
Run:  python3 pipeline2.py <dataset_dir> <out_json>
"""
import json, sys, warnings, math
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score, roc_auc_score
import xgboost as xgb

warnings.filterwarnings("ignore")
DS, OUT = sys.argv[1], sys.argv[2]
H, SEG, SEED = 6, 100.0, 7
BANDS = [150.0, 500.0, 1000.0]          # Edwards & Hosein (2025): normal / minor / major / critical
EXCESS_DWELL_EVENT = 30.0               # seconds above the local 90th-percentile dwell (design choice)

man = json.load(open(f"{DS}/manifest.json"))
proj = json.load(open(f"{DS}/project.json"))["routes"][0]
LAT0, LON0 = 18.056, 120.556
KX, KY = 111320 * np.cos(np.radians(LAT0)), 110574.0
to_xy = lambda lat, lon: ((np.asarray(lon) - LON0) * KX, (np.asarray(lat) - LAT0) * KY)

def polyline(coords):
    c = np.array(coords); x, y = to_xy(c[:, 1], c[:, 0]); P = np.c_[x, y]
    return P, np.r_[0, np.cumsum(np.hypot(*np.diff(P, axis=0).T))]
PATH = {"outbound": polyline(proj["paths"]["outbound"]["coords"]),
        "return": polyline(proj["paths"]["return"]["coords"])}

def project(P, cum, px, py):
    A, B = P[:-1], P[1:]; AB = B - A; L2 = (AB ** 2).sum(1) + 1e-9
    s = np.zeros(len(px)); d = np.zeros(len(px)); brg = np.zeros(len(px))
    for i, (x, y) in enumerate(zip(px, py)):
        t = np.clip(((x - A[:, 0]) * AB[:, 0] + (y - A[:, 1]) * AB[:, 1]) / L2, 0, 1)
        dd = np.hypot(x - (A[:, 0] + t * AB[:, 0]), y - (A[:, 1] + t * AB[:, 1])); k = dd.argmin()
        s[i] = cum[k] + t[k] * np.sqrt(L2[k]); d[i] = dd[k]; brg[i] = math.atan2(AB[k, 1], AB[k, 0])
    return s, d, brg

wrap = lambda a: (a + np.pi) % (2 * np.pi) - np.pi

# ----------------------------------------------------------------------------- frames
def make_frame(tid, t, x, y, key):
    """t: DatetimeIndex(UTC). Everything here is independent of any training set."""
    dt = np.r_[10.0, np.diff(t.asi8 / 1e9)]
    step = np.r_[0, np.hypot(np.diff(x), np.diff(y))]
    v = step / dt; a = np.r_[0, np.diff(v)] / dt; jerk = np.r_[0, np.diff(a)] / dt
    hd = np.arctan2(np.r_[0, np.diff(y)], np.r_[0, np.diff(x)])
    turn = np.abs(np.r_[0, wrap(np.diff(hd))]) / dt; turn[v < 0.5] = 0
    P, cum = PATH[key]; s, d, brg = project(P, cum, x, y)
    hdev = np.degrees(np.abs(wrap(hd - brg))); hdev[v < 1.0] = 0
    if key == "return": s = cum[-1] - s
    loc = t.tz_convert("Asia/Manila")
    f = pd.DataFrame(dict(trip=tid, dir=int(key == "return"), t=t, x=x, y=y, v=v, a=a, jerk=jerk, turn=turn,
                          s=s, d=d, heading_dev=hdev, hour=loc.hour + loc.minute / 60, dow=loc.dayofweek,
                          elapsed=(t - t[0]).total_seconds().values))
    f["stopped"] = (f.v < 0.5).astype(int)
    f["v_mean6"] = f.v.rolling(6, min_periods=1).mean()
    f["v_std6"] = f.v.rolling(6, min_periods=2).std().fillna(0)
    f["a_min6"] = f.a.rolling(6, min_periods=1).min()
    f["a_absmax6"] = f.a.abs().rolling(6, min_periods=1).max()
    f["stops6"] = f.stopped.rolling(6, min_periods=1).sum()
    f["turn_sum6"] = f.turn.rolling(6, min_periods=1).sum()
    run = np.zeros(len(f)); c = 0
    for i, z in enumerate(f.stopped): c = c + 1 if z else 0; run[i] = c
    f["dwell"] = run * 10
    f["cum_stop"] = f.stopped.cumsum() * 10
    f["sbin"] = (f.s // SEG).astype(int).clip(lower=0)
    return f

recs = man["recordings"]
RAW = {}
for i, r in enumerate(recs):
    df = pd.read_csv(f"{DS}/{r['file']}"); t = pd.DatetimeIndex(pd.to_datetime(df.timestamp, utc=True))
    x, y = to_xy(df.latitude, df.longitude)
    RAW[i] = dict(t=t, x=np.asarray(x), y=np.asarray(y), key="outbound" if r["direction"] == "outbound" else "return")
FR = {i: make_frame(i, v["t"], v["x"], v["y"], v["key"]) for i, v in RAW.items()}
dep = pd.Series({i: pd.to_datetime(r["departure"]) for i, r in enumerate(recs)}).sort_values()
order = list(dep.index); n_trips = len(order)
NB = int(max(f.sbin.max() for f in FR.values())) + 1

# ----------------------------------------------------------------------------- context (training-set dependent)
class Ctx: pass
def fit_ctx(train_ids):
    Ft = pd.concat([FR[i] for i in train_ids], ignore_index=True); Ft["part"] = (Ft.hour >= 12).astype(int)
    c = Ctx(); c.train = set(train_ids)
    c.base = Ft.groupby(["sbin", "dir", "part"]).v.median().to_dict(); c.gmed = float(Ft.v.median())
    c.dref = Ft.groupby(["sbin", "dir"]).dwell.quantile(0.9).to_dict()
    c.pstop = Ft.groupby(["sbin", "dir"]).stopped.mean().to_dict()
    c.hist = {}
    for dr in (0, 1):
        g = Ft[Ft.dir == dr]; c.hist[dr] = (np.c_[g.x, g.y], g.trip.values)
    return c

def apply_ctx(F, c):
    F = F.reset_index(drop=True).copy(); F["part"] = (F.hour >= 12).astype(int)
    F["v_base"] = [c.base.get(k, c.gmed) for k in zip(F.sbin, F.dir, F.part)]
    F["v_resid"] = F.v - F.v_base
    F["dwell_ref"] = [c.dref.get(k, 0.0) for k in zip(F.sbin, F.dir)]
    F["excess_dwell"] = (F.dwell - F.dwell_ref).clip(lower=0)
    F["stop_unexp"] = F.stopped * (1 - np.array([c.pstop.get(k, 0.0) for k in zip(F.sbin, F.dir)]))
    dh = np.zeros(len(F))
    for tid, g in F.groupby("trip"):
        Hp, ht = c.hist[int(g.dir.iloc[0])]; m = ht != tid
        P = np.c_[g.x, g.y]
        dh[g.index] = np.sqrt(((P[:, None, :] - Hp[m][None, :, :]) ** 2).sum(2)).min(1) if m.any() else 9e3
    F["d_hist"] = dh
    F["band_hist"] = np.digitize(F.d_hist, BANDS, right=True)
    F["band_poly"] = np.digitize(F.d, BANDS, right=True)
    dr = np.zeros(len(F))
    for tid, g in F.groupby("trip"):
        run = 0
        for i, b in zip(g.index, np.maximum(g.band_hist, g.band_poly).values):
            run = run + 10 if b >= 1 else 0; dr[i] = run
    F["dev_run"] = dr
    return F

V1 = ["v", "a", "jerk", "turn", "d", "v_resid", "v_std6", "dwell"]
V2 = V1 + ["heading_dev", "excess_dwell", "d_hist"]
BASE = ["v", "a", "jerk", "turn", "d", "v_mean6", "v_std6", "a_min6", "a_absmax6", "stops6", "dwell", "turn_sum6",
        "s", "dir", "hour", "dow", "elapsed", "v_resid"]
DEVG = ["d_hist", "band_hist", "band_poly", "heading_dev", "dev_run"]
DWG = ["excess_dwell", "stop_unexp", "cum_stop"]
GROUPS = {"v1 features": BASE, "+ deviation group": BASE + DEVG, "+ dwell group": BASE + DWG,
          "v2 full (+deviation +dwell)": BASE + DEVG + DWG}

class RotIF:
    """Randomly rotated Isolation Forests, averaged. Addresses the axis-parallel split artefact (Hariri et al.)."""
    def __init__(self, k=8, seed=SEED): self.k, self.seed = k, seed
    def fit(self, X):
        X = np.asarray(X, float); self.mu, self.sd = X.mean(0), X.std(0) + 1e-9
        rng = np.random.RandomState(self.seed); Z = (X - self.mu) / self.sd; self.R, self.m = [], []
        for j in range(self.k):
            Q, _ = np.linalg.qr(rng.randn(X.shape[1], X.shape[1])); self.R.append(Q)
            self.m.append(IsolationForest(n_estimators=60, max_samples=256, random_state=self.seed + j).fit(Z @ Q))
        return self
    def score(self, X):
        Z = (np.asarray(X, float) - self.mu) / self.sd
        return np.mean([-m.score_samples(Z @ Q) for m, Q in zip(self.m, self.R)], axis=0)

def ecdf(ref):
    ref = np.sort(ref); return lambda s: np.searchsorted(ref, s, side="right") / len(ref)

def fwd(F, ev):
    out = np.zeros(len(F), int)
    for tid, g in F.groupby("trip", sort=False):
        e = g[ev].values; n = len(e)
        out[g.index.values] = [e[i + 1:i + 1 + H].max() if i + 1 < n else 0 for i in range(n)]
    return out

def fit_xgb(Xtr, ytr, Xva, yva, seed=SEED):
    m = xgb.XGBClassifier(n_estimators=400, learning_rate=0.04, max_depth=3, subsample=0.8, colsample_bytree=0.8,
                          min_child_weight=3, reg_lambda=2.0, eval_metric="aucpr", early_stopping_rounds=40,
                          random_state=seed, n_jobs=2)
    m.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False); return m

def safe_pr(y, p): return average_precision_score(y, p) if y.sum() > 0 else np.nan
def safe_roc(y, p): return roc_auc_score(y, p) if 0 < y.sum() < len(y) else np.nan

# ----------------------------------------------------------------------------- injections
def inject(raw, kind, rng, c):
    t, x, y, key = raw["t"], raw["x"].copy(), raw["y"].copy(), raw["key"]; n = len(x)
    mask = np.zeros(n, bool)
    if kind.startswith("detour"):
        peak = {"detour_minor": 300., "detour_major": 750., "detour_critical": 1200.}[kind]
        r = int(max(3, math.ceil(peak / 150))); L = 2 * r + 1
        if n < L + 10: return None
        i0 = int(rng.randint(4, n - L - 4)); hd = np.arctan2(y[i0 + r] - y[i0], x[i0 + r] - x[i0])
        nx, ny = -math.sin(hd), math.cos(hd); sg = rng.choice([-1, 1])
        for j in range(L):
            tri = 1 - abs(j - r) / (r + 1); x[i0 + j] += sg * peak * tri * nx; y[i0 + j] += sg * peak * tri * ny
        mask[i0:i0 + L] = True
        return dict(t=t, x=x, y=y, key=key), mask, i0
    return None

def inject_stop(raw, kind, rng, F_clean, c):
    k = 9 if kind == "stop_90s" else 18; n = len(raw["x"])
    ps = np.array([c.pstop.get((int(b), int(F_clean.dir.iloc[0])), 0.0) for b in F_clean.sbin])
    cand = [i for i in range(5, n - 5) if ps[i] < 0.05] or list(range(5, n - 5))
    i0 = int(rng.choice(cand))
    x = np.r_[raw["x"][:i0 + 1], raw["x"][i0] + rng.randn(k) * 1.5, raw["x"][i0 + 1:]]
    y = np.r_[raw["y"][:i0 + 1], raw["y"][i0] + rng.randn(k) * 1.5, raw["y"][i0 + 1:]]
    t = pd.date_range(raw["t"][0], periods=len(x), freq="10s")
    mask = np.zeros(len(x), bool); mask[i0 + 1:i0 + 1 + k] = True
    return dict(t=t, x=x, y=y, key=raw["key"]), mask, i0 + 1

def inject_speed(raw, kind, rng):
    f = 1.5 if kind == "speed_up" else 0.5; n = len(raw["x"]); W = 8
    if n < W + 10: return None
    c = np.r_[0, np.cumsum(np.hypot(np.diff(raw["x"]), np.diff(raw["y"])))] + np.arange(n) * 1e-6
    i0 = int(rng.randint(3, n - W - 3)); tgt = c.copy()
    tgt[i0:i0 + W] = c[i0] + f * (c[i0:i0 + W] - c[i0])
    tgt[i0 + W:] = c[i0 + W:] + (f - 1) * (c[i0 + W - 1] - c[i0])
    tgt = np.clip(tgt, 0, c[-1])
    x = np.interp(tgt, c, raw["x"]); y = np.interp(tgt, c, raw["y"])
    mask = np.zeros(n, bool); mask[i0:i0 + W] = True
    return dict(t=raw["t"], x=x, y=y, key=raw["key"]), mask, i0

KINDS = ["detour_minor", "detour_major", "detour_critical", "stop_90s", "stop_180s", "speed_up", "crawl"]

def build_injected(test_ids, c, seed):
    rng = np.random.RandomState(seed); out = []; uid = 10000 + seed * 1000
    for tid in test_ids:
        for kind in KINDS:
            if kind.startswith("detour"): res = inject(RAW[tid], kind, rng, c)
            elif kind.startswith("stop"): res = inject_stop(RAW[tid], kind, rng, FR[tid], c)
            else: res = inject_speed(RAW[tid], kind, rng)
            if res is None: continue
            raw, mask, ws = res; uid += 1
            out.append((kind, tid, mask, ws, make_frame(uid, raw["t"], raw["x"], raw["y"], raw["key"])))
    return out

# ----------------------------------------------------------------------------- one fold
def run_fold(tr_ids, va_ids, te_ids, Q=0.10, injection=False, seed=SEED, want=False):
    c = fit_ctx(tr_ids)
    D = apply_ctx(pd.concat([FR[i] for i in tr_ids + va_ids + te_ids], ignore_index=True), c)
    tr = D.trip.isin(tr_ids).values
    if1 = IsolationForest(n_estimators=300, max_samples=256, random_state=seed).fit(D.loc[tr, V1])
    if2 = IsolationForest(n_estimators=300, max_samples=256, random_state=seed).fit(D.loc[tr, V2])
    rot = RotIF(seed=seed).fit(D.loc[tr, V2])
    D["if1"] = -if1.score_samples(D[V1]); D["if2"] = -if2.score_samples(D[V2]); D["rot"] = rot.score(D[V2])
    tau = np.quantile(D.loc[tr, "if2"], 1 - Q)
    D["ev_if"] = (D.if2 >= tau).astype(int)
    D["ev_rule"] = ((np.maximum(D.band_hist, D.band_poly) >= 1) | (D.excess_dwell >= EXCESS_DWELL_EVENT)).astype(int)
    D["ev_any"] = np.maximum(D.ev_if, D.ev_rule)
    res = dict(base_rate={}, rows={})
    va = D.trip.isin(va_ids).values; te = D.trip.isin(te_ids).values
    models = {}
    for tname, ev in (("IF event", "ev_if"), ("Rule event (deviation / excess dwell)", "ev_rule"), ("Either", "ev_any")):
        y = fwd(D, ev); res["base_rate"][tname] = float(y[te].mean())
        rows = {}
        rows["Persistence (event now → event soon)"] = (safe_pr(y[te], D.loc[te, ev].values), safe_roc(y[te], D.loc[te, ev].values))
        for gname, cols in GROUPS.items():
            if y[tr].sum() < 3 or y[va].sum() < 1: rows[gname] = (np.nan, np.nan); continue
            m = fit_xgb(D.loc[tr, cols], y[tr], D.loc[va, cols], y[va], seed)
            p = m.predict_proba(D.loc[te, cols])[:, 1]; rows[gname] = (safe_pr(y[te], p), safe_roc(y[te], p))
            if ev == "ev_if" and gname == "v2 full (+deviation +dwell)": models["xgb"] = (m, cols)
        res["rows"][tname] = {k: dict(prauc=v[0], roc=v[1]) for k, v in rows.items()}
    res["tau"] = float(tau)
    if not injection: return (res, D, c, models) if want else res
    # ---- injected-anomaly benchmark (exact ground truth, independent of every training label)
    m, cols = models["xgb"]
    ref = {k: ecdf(D.loc[tr, k].values) for k in ("if1", "if2", "rot", "d_hist", "excess_dwell", "d")}
    ref["xgb"] = ecdf(m.predict_proba(D.loc[tr, cols])[:, 1])
    def detectors(F):
        s = {k: ref[k](F[k].values) for k in ("if1", "if2", "rot", "d_hist", "excess_dwell", "d")}
        s["xgb"] = ref["xgb"](m.predict_proba(F[cols])[:, 1])
        s["fusion"] = np.maximum.reduce([s["if2"], s["d_hist"], s["excess_dwell"]])
        return s
    clean = D[te]; neg = detectors(clean)
    inj = build_injected(te_ids, c, seed)
    store = []
    for kind, tid, mask, ws, Fi in inj:
        Fi = apply_ctx(Fi, c); Fi['if1'] = -if1.score_samples(Fi[V1]); Fi['if2'] = -if2.score_samples(Fi[V2]); Fi['rot'] = rot.score(Fi[V2])
        s = detectors(Fi)
        lead = np.zeros(len(Fi), bool); lead[max(0, ws - H):ws] = True; lead &= ~mask
        store.append((kind, {k: v[mask] for k, v in s.items()}, {k: v[lead] for k, v in s.items()}, mask.sum(), Fi, s, mask))
    out = dict(res=res, neg=neg, inj=store)
    return (out, D, c, models) if want else out

def tpr_at_fpr(neg, pos, fpr=0.05):
    thr = np.quantile(neg, 1 - fpr); return float((pos > thr).mean()) if len(pos) else np.nan

# ----------------------------------------------------------------------------- 1) rolling-origin folds
folds = [(k - 6 if k - 6 > 0 else k, order[:k - 6], order[k - 6:k], order[k:k + 6]) for k in (24, 30, 36, 42, 48, 54)]
fold_out = []; DET = ["if1", "if2", "rot", "d_hist", "excess_dwell", "d", "xgb", "fusion"]
pool_neg = {k: [] for k in DET}; pool_pos = {k: {d: [] for d in DET} for k in KINDS}; pool_lead = {k: {d: [] for d in DET} for k in KINDS}
demo = None
for fi, (_, trn, van, ten) in enumerate(folds):
    o, D, c, mdl = run_fold(trn, van, ten, injection=True, want=True)
    fold_out.append(o["res"])
    for d in DET: pool_neg[d].append(o["neg"][d])
    for kind, pos, lead, nn, Fi, s, mask in o["inj"]:
        for d in DET: pool_pos[kind][d].append(pos[d]); pool_lead[kind][d].append(lead[d])
        if fi == 4 and kind == "detour_major" and demo is None:
            demo = dict(kind=kind, t=[int(v) for v in Fi.elapsed], speed_kmh=[round(float(v * 3.6), 1) for v in Fi.v],
                        mask=[int(v) for v in mask], band=[int(v) for v in np.maximum(Fi.band_hist, Fi.band_poly)],
                        d=[round(float(v), 1) for v in Fi.d], **{k: [round(float(v), 3) for v in s[k]] for k in ("if2", "xgb", "fusion", "d_hist", "excess_dwell")})
        if fi == 4 and kind == "stop_180s":
            demo_stop = dict(kind=kind, t=[int(v) for v in Fi.elapsed], speed_kmh=[round(float(v * 3.6), 1) for v in Fi.v], mask=[int(v) for v in mask],
                             **{k: [round(float(v), 3) for v in s[k]] for k in ("if2", "xgb", "fusion", "d_hist", "excess_dwell")})
neg_all = {d: np.concatenate(pool_neg[d]) for d in DET}
inj_tab = {}
for kind in KINDS:
    inj_tab[kind] = {}
    for d in DET:
        pos = np.concatenate(pool_pos[kind][d]); ld = np.concatenate(pool_lead[kind][d]) if pool_lead[kind][d] else np.array([])
        y = np.r_[np.zeros(len(neg_all[d])), np.ones(len(pos))]; sc = np.r_[neg_all[d], pos]
        yl = np.r_[np.zeros(len(neg_all[d])), np.ones(len(ld))]; sl = np.r_[neg_all[d], ld]
        inj_tab[kind][d] = dict(auc=safe_roc(y, sc), tpr5=tpr_at_fpr(neg_all[d], pos), lead_auc=safe_roc(yl, sl) if len(ld) else np.nan,
                                n_pos=int(len(pos)), n_lead=int(len(ld)))
n_neg = int(len(neg_all["if2"]))

# fold summary
def fsum(tname, g, key="prauc"):
    v = [f["rows"][tname][g][key] for f in fold_out]; return float(np.nanmean(v)), float(np.nanstd(v))
abl = {t: {g: dict(prauc=fsum(t, g)[0], prauc_sd=fsum(t, g)[1], roc=fsum(t, g, "roc")[0]) for g in fold_out[0]["rows"][t]} for t in fold_out[0]["rows"]}
base_rates = {t: float(np.mean([f["base_rate"][t] for f in fold_out])) for t in fold_out[0]["base_rate"]}

# ----------------------------------------------------------------------------- 2) sensitivity to the event threshold
sens = []
for Q in (0.05, 0.10, 0.15, 0.20):
    pr, br, ps = [], [], []
    for _, trn, van, ten in folds:
        o = run_fold(trn, van, ten, Q=Q)
        pr.append(o["rows"]["IF event"]["v2 full (+deviation +dwell)"]["prauc"]); ps.append(o["rows"]["IF event"]["Persistence (event now → event soon)"]["prauc"])
        br.append(o["base_rate"]["IF event"])
    sens.append(dict(Q=Q, base=float(np.mean(br)), xgb=float(np.nanmean(pr)), persist=float(np.nanmean(ps)),
                     lift=float(np.nanmean(pr) / np.mean(br))))

# ----------------------------------------------------------------------------- 3) isolation-forest stability
_, trn, van, ten = folds[3]
c = fit_ctx(trn); D = apply_ctx(pd.concat([FR[i] for i in trn + van + ten], ignore_index=True), c); trm = D.trip.isin(trn).values
sets, scs = {}, {}
for s_ in range(5):
    m = IsolationForest(n_estimators=300, max_samples=256, random_state=s_).fit(D.loc[trm, V2]); sc = -m.score_samples(D[V2])
    scs[("seed", s_)] = sc; sets[("seed", s_)] = set(np.where(sc >= np.quantile(sc, .9))[0])
for ne in (100, 300, 600):
    m = IsolationForest(n_estimators=ne, max_samples=256, random_state=0).fit(D.loc[trm, V2]); sc = -m.score_samples(D[V2])
    scs[("trees", ne)] = sc; sets[("trees", ne)] = set(np.where(sc >= np.quantile(sc, .9))[0])
jac = lambda a, b: len(a & b) / len(a | b)
sk = [k for k in sets if k[0] == "seed"]; tk = [k for k in sets if k[0] == "trees"]
stab = dict(seed_jaccard=float(np.mean([jac(sets[a], sets[b]) for i, a in enumerate(sk) for b in sk[i + 1:]])),
            seed_spearman=float(np.mean([spearmanr(scs[a], scs[b])[0] for i, a in enumerate(sk) for b in sk[i + 1:]])),
            trees_jaccard=float(np.mean([jac(sets[a], sets[b]) for i, a in enumerate(tk) for b in tk[i + 1:]])),
            trees_spearman=float(np.mean([spearmanr(scs[a], scs[b])[0] for i, a in enumerate(tk) for b in tk[i + 1:]])))

# ----------------------------------------------------------------------------- 4) cold-start curve
cold = []; test_block = order[48:60]
for N in (6, 12, 24, 36, 48):
    pr, auc_f, auc_if = [], [], []
    for rep in range(3):
        rng = np.random.RandomState(rep)
        ids = list(order[:48]) if N == 48 else sorted(rng.choice(order[:48], N, replace=False).tolist(), key=lambda i: dep[i])
        cut = max(2, int(len(ids) * .75)); trn, van = ids[:cut], ids[cut:]
        o = run_fold(trn, van, test_block, injection=True, seed=SEED + rep)
        pr.append(o["res"]["rows"]["IF event"]["v2 full (+deviation +dwell)"]["prauc"])
        auc_f.append(np.nanmean([tpr_at_fpr(o["neg"]["fusion"], np.concatenate([s_[1]["fusion"] for s_ in o["inj"] if s_[0] == kk0])) for kk0 in KINDS]))
        auc_if.append(np.nanmean([tpr_at_fpr(o["neg"]["if2"], np.concatenate([s_[1]["if2"] for s_ in o["inj"] if s_[0] == kk0])) for kk0 in KINDS]))
    cold.append(dict(N=N, prauc=float(np.nanmean(pr)), sd=float(np.nanstd(pr)), fusion_tpr=float(np.nanmean(auc_f)), if_tpr=float(np.nanmean(auc_if))))

# ----------------------------------------------------------------------------- 5) data facts for deviation bands (all trips, descriptive)
call = fit_ctx(order); Dall = apply_ctx(pd.concat([FR[i] for i in order], ignore_index=True), call)
bh = np.bincount(Dall.band_hist, minlength=4).tolist(); bp = np.bincount(Dall.band_poly, minlength=4).tolist()
step_med = float((Dall.v * 10).median())
facts = dict(band_hist=bh, band_poly=bp, n=int(len(Dall)), max_d_poly=float(Dall.d.max()), max_d_hist=float(Dall.d_hist.max()),
             median_step_m=step_med, ev_rule_rate=float(((np.maximum(Dall.band_hist, Dall.band_poly) >= 1) | (Dall.excess_dwell >= EXCESS_DWELL_EVENT)).mean()),
             frac_band_poly_ge1=float((Dall.band_poly >= 1).mean()), frac_band_hist_ge1=float((Dall.band_hist >= 1).mean()),
             frac_excess30=float((Dall.excess_dwell >= EXCESS_DWELL_EVENT).mean()))

res = dict(abl=abl, base_rates=base_rates, folds=len(folds), fold_sizes=[(len(a), len(b), len(c_)) for _, a, b, c_ in folds],
           inj=inj_tab, n_neg=n_neg, sens=sens, stab=stab, cold=cold, facts=facts, demo=demo, demo_stop=demo_stop,
           params=dict(H=H, bands=BANDS, excess_dwell=EXCESS_DWELL_EVENT, V1=V1, V2=V2, kinds=KINDS))
json.dump(res, open(OUT, "w"), indent=1, default=float)
print(json.dumps(dict(abl=abl, base_rates=base_rates, sens=sens, stab=stab, cold=cold, facts=facts), indent=1, default=float))
for k in KINDS:
    print(k, {d: (round(v["auc"], 2), round(v["tpr5"], 2), None if v["lead_auc"] != v["lead_auc"] else round(v["lead_auc"], 2)) for d, v in inj_tab[k].items()})
