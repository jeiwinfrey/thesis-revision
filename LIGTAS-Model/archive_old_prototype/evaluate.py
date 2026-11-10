"""
LIGTAS evaluation tool (v2).  Verifies accuracy on a dataset folder and writes an HTML + JSON report.

  python3 evaluate.py <dataset_dir> [--labels labels.csv] [--out-dir DIR] [--alert-level 2] [--max-false-alerts 1.0] [--budget 1.0]

Checks, each with a fixed pass/fail gate:
  0  data quality of the CSVs
  1  forecast accuracy against a persistence baseline (rolling-origin folds; a trip is never split)
  2  detection of planted events: designed-for, speed changes (seen and HELD-OUT magnitudes) and event types the rules were NOT designed for,
     against rules, LOF, One-Class SVM, max-z, the forest, an extended forest, a supervised detector trained on planted events, and a hybrid
  3  false alarms: fixed concern levels, and a budgeted alert policy (threshold chosen on validation trips to meet a false-alarm budget, with debounce)
  4  calibration and conformal coverage
  5  real labels (optional)
labels.csv columns:  file (e.g. a-to-b/trip-0001.csv) OR trip_id ; incident (0/1) ; optional incident_time (ISO 8601)
"""
import sys, argparse, os, html as html_mod
ap = argparse.ArgumentParser()
ap.add_argument("dataset"); ap.add_argument("--labels"); ap.add_argument("--out-dir", default=os.path.dirname(os.path.abspath(__file__)))
ap.add_argument("--alert-level", type=int, default=2); ap.add_argument("--max-false-alerts", type=float, default=1.0); ap.add_argument("--budget", type=float, default=1.0)
ARGS = ap.parse_args()
HERE = os.path.dirname(os.path.abspath(__file__))
src = open(os.path.join(HERE, "pipeline3.py")).read()
head = src[:src.index("# ------------------------------------------------------------------ 1) rolling-origin folds")]
sys.argv = ["pipeline3", ARGS.dataset, "/dev/null", "/dev/null"]
exec(compile(head, "pipeline3_head", "exec"))        # model definitions (features, forests, injections, etc.)

from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
import time as _time
T0 = _time.time()

# ============================================================== extra features for speed / erratic motion (added on top of pipeline3)
def add_speed_feats(f):
    f = f.copy(); mv = f.v.where(f.v > 0.5)
    last3 = mv.rolling(3, min_periods=1).mean(); prev6 = mv.shift(3).rolling(6, min_periods=2).mean()
    f["v_ratio"] = np.log((last3 + 1) / (prev6 + 1)).fillna(0).clip(-2, 2)
    sg = np.sign(f.a.values); sc = np.r_[0, (sg[1:] * sg[:-1] < 0).astype(int)]
    f["a_sign6"] = pd.Series(sc, index=f.index).rolling(6, min_periods=1).sum()
    f["d_absdiff4"] = f.d.diff().abs().rolling(4, min_periods=1).mean().fillna(0)
    f["hdev_hi3"] = (f.heading_dev > 120).astype(int).rolling(3, min_periods=1).sum()
    return f
_make_frame_old = make_frame; _apply_ctx_old = apply_ctx
def make_frame(tid, t, x, y, key): return add_speed_feats(_make_frame_old(tid, t, x, y, key))
def apply_ctx(F, c):
    F = _apply_ctx_old(F, c); mz = F.v_z.where(F.v > 0.5)
    F["v_zm5"] = mz.groupby(F.trip).transform(lambda s: s.rolling(5, min_periods=1).mean()).fillna(0); return F
FR = {i: add_speed_feats(f) for i, f in FR.items()}
SPD = ["v_ratio", "v_zm5", "a_sign6", "d_absdiff4", "hdev_hi3"]
V4 = V3 + SPD
F_ASD = ["v", "a", "jerk", "turn", "d", "v_resid", "v_std6", "dwell", "heading_dev", "excess_dwell", "d_hist", "v_z", "v_zm5", "v_trend3", "d_rate3", "time_since_stop",
         "v_mean6", "a_absmax6", "turn_sum6", "stops6", "dev_run", "band_hist", "band_poly", "v_ratio", "a_sign6", "d_absdiff4", "hdev_hi3", "v_rel", "a_mean3", "hdev_rate"]

# ============================================================== 0) data quality
def data_checks():
    issues, per = [], []
    for i, r in enumerate(recs):
        df = pd.read_csv(f"{DS}/{r['file']}"); need = {"timestamp", "latitude", "longitude"}
        if not need <= set(df.columns): issues.append((r["file"], "missing columns")); continue
        t = pd.to_datetime(df.timestamp, utc=True, errors="coerce"); bad = int(t.isna().sum() + df.latitude.isna().sum() + df.longitude.isna().sum())
        dt = t.diff().dt.total_seconds().dropna(); x, y = to_xy(df.latitude, df.longitude); sp = np.hypot(np.diff(x), np.diff(y)) / np.maximum(dt.values, 1e-6) * 3.6 if len(dt) else np.array([0])
        row = dict(file=r["file"], n=len(df), median_dt=float(dt.median()) if len(dt) else np.nan, max_gap=float(dt.max()) if len(dt) else np.nan,
                   dup=int(t.duplicated().sum()), nonmono=int((dt <= 0).sum()), teleports=int((sp > 200).sum()), bad=bad); per.append(row)
        for k, cond in (("unparseable/missing values", bad > 0), ("duplicate timestamps", row["dup"] > 0), ("timestamps not increasing", row["nonmono"] > 0),
                        ("speed above 200 km/h between fixes", row["teleports"] > 0), ("fewer than 20 fixes", len(df) < 20), ("gap above 60 s", row["max_gap"] > 60)):
            if cond: issues.append((r["file"], k))
    P = pd.DataFrame(per)
    return dict(n_trips=len(P), n_fixes=int(P.n.sum()), median_dt=float(P.median_dt.median()), max_gap=float(P.max_gap.max()), issues=issues,
                n_issue_trips=len(set(f for f, _ in issues)), critical=any(k in ("missing columns", "unparseable/missing values", "timestamps not increasing") for _, k in issues))
DQ = data_checks()

# ============================================================== planted events
def ins_rows(raw, i0, xs, ys):
    x = np.r_[raw["x"][:i0 + 1], xs, raw["x"][i0 + 1:]]; y = np.r_[raw["y"][:i0 + 1], ys, raw["y"][i0 + 1:]]
    t = pd.date_range(raw["t"][0], periods=len(x), freq="10s"); m = np.zeros(len(x), bool); m[i0 + 1:i0 + 1 + len(xs)] = True
    return dict(t=t, x=x, y=y, key=raw["key"]), m, i0 + 1
def inj_loop(raw, rng, R=110.0, L=10):
    n = len(raw["x"]); i0 = int(rng.randint(5, n - 6)); hd = np.arctan2(raw["y"][i0 + 1] - raw["y"][i0], raw["x"][i0 + 1] - raw["x"][i0]); sg = rng.choice([-1, 1])
    nx, ny = -math.sin(hd) * sg, math.cos(hd) * sg; cx, cy = raw["x"][i0] + R * nx, raw["y"][i0] + R * ny
    ang = math.atan2(-ny, -nx) + sg * np.linspace(0, 2 * np.pi, L + 2)[1:-1]; return ins_rows(raw, i0, cx + R * np.cos(ang), cy + R * np.sin(ang))
def inj_stop_and_go(raw, rng, rows=12):
    n = len(raw["x"]); i0 = int(rng.randint(5, n - 6)); p0 = np.array([raw["x"][i0], raw["y"][i0]]); p1 = np.array([raw["x"][i0 + 1], raw["y"][i0 + 1]])
    u = (p1 - p0) / (np.linalg.norm(p1 - p0) + 1e-9); pos = []; adv = 0.0
    for step in [0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1][:rows]: adv += 12.0 * step; pos.append(p0 + u * min(adv, np.linalg.norm(p1 - p0) * 0.9))
    pos = np.array(pos); return ins_rows(raw, i0, pos[:, 0] + rng.randn(len(pos)), pos[:, 1] + rng.randn(len(pos)))
def inj_reversal(raw, rng):
    n = len(raw["x"]); i0 = int(rng.randint(6, n - 6)); idx = [i0 - 1, i0 - 2, i0 - 3, i0 - 2, i0 - 1]; return ins_rows(raw, i0, raw["x"][idx], raw["y"][idx])
def inj_zigzag(raw, rng, W=8, amp=45.0):
    t, x, y = raw["t"], raw["x"].copy(), raw["y"].copy(); n = len(x); i0 = int(rng.randint(4, n - W - 4))
    hd = np.arctan2(y[i0 + W] - y[i0], x[i0 + W] - x[i0]); nx, ny = -math.sin(hd), math.cos(hd); mask = np.zeros(n, bool)
    for j in range(W): sgn = 1 if j % 2 == 0 else -1; x[i0 + j] += sgn * amp * nx; y[i0 + j] += sgn * amp * ny; mask[i0 + j] = True
    return dict(t=t, x=x, y=y, key=raw["key"]), mask, i0
def inj_smooth_detour(raw, rng, peak=500.0, L=16):
    t, x, y = raw["t"], raw["x"].copy(), raw["y"].copy(); n = len(x)
    if n < L + 10: return None
    i0 = int(rng.randint(4, n - L - 4)); hd = np.arctan2(y[i0 + L // 2] - y[i0], x[i0 + L // 2] - x[i0]); nx, ny = -math.sin(hd), math.cos(hd); sg = rng.choice([-1, 1]); off = np.zeros(n)
    for j in range(L):
        u = j / (L - 1); w = 2 * min(u, 1 - u); off[i0 + j] = peak * (3 * w ** 2 - 2 * w ** 3); x[i0 + j] += sg * off[i0 + j] * nx; y[i0 + j] += sg * off[i0 + j] * ny
    mask = off > 150; return dict(t=t, x=x, y=y, key=raw["key"]), mask, int(np.argmax(mask))

def inj_speed_profile(raw, mult, rng):
    n = len(raw["x"]); W = len(mult)
    if n < W + 10: return None
    c0 = np.r_[0, np.cumsum(np.hypot(np.diff(raw["x"]), np.diff(raw["y"])))] + np.arange(n) * 1e-6
    i0 = int(rng.randint(3, n - W - 3)); dl = np.diff(c0).copy(); dl[i0:i0 + W] *= mult
    c2 = np.clip(np.r_[0, np.cumsum(dl)], 0, c0[-1]); mask = np.zeros(n, bool); mask[i0 + 1:i0 + W + 1] = True
    return dict(t=raw["t"], x=np.interp(c2, c0, raw["x"]), y=np.interp(c2, c0, raw["y"]), key=raw["key"]), mask, i0 + 1

GROUPS_ORDER = ["designed", "speed_seen", "speed_heldout", "speed_newtype", "unseen"]
GTITLE = {"designed": "Events the rules were designed for", "speed_seen": "Speed changes, magnitudes used in training", "speed_heldout": "Speed changes, magnitudes NEVER seen in training", "speed_newtype": "Speed events of a NEW KIND (jitter, brake pulse), never trained on", "unseen": "Event types the rules were NOT designed for (and never trained on)"}
KIND_GROUP = {"detour_minor": "designed", "detour_major": "designed", "detour_critical": "designed", "smooth_detour_500": "designed", "stop_90s": "designed", "stop_empirical": "designed",
              "speed_up": "speed_seen", "crawl": "speed_seen", "speed_1.25": "speed_heldout", "speed_1.8": "speed_heldout", "crawl_0.7": "speed_heldout", "crawl_0.3": "speed_heldout",
              "speed_jitter": "speed_newtype", "brake_pulse": "speed_newtype", "loop": "unseen", "stop_and_go": "unseen", "reversal": "unseen", "zigzag": "unseen"}
KLAB = {"detour_minor": "Detour 300 m", "detour_major": "Detour 750 m", "detour_critical": "Detour 1,200 m", "smooth_detour_500": "Smooth detour 500 m", "stop_90s": "Stop 90 s",
        "stop_empirical": "Stop, duration from data", "speed_up": "Speed ×1.5", "crawl": "Crawl ×0.5", "speed_1.25": "Speed ×1.25", "speed_1.8": "Speed ×1.8", "crawl_0.7": "Crawl ×0.7",
        "crawl_0.3": "Crawl ×0.3", "speed_jitter": "Speed jitter (±50% per fix)", "brake_pulse": "Brake pulse (3 fixes at 20% speed)", "loop": "Loop (110 m radius)", "stop_and_go": "Stop-and-go (20 s stops)", "reversal": "Backtracking", "zigzag": "Zigzag (±45 m)"}
EKINDS = list(KIND_GROUP)
TRAIN_KINDS = ["detour_minor", "detour_major", "detour_critical", "stop_90s", "speed_up", "crawl"]   # the supervised detector only ever sees these

def make_inj(name, tid, rng, c, F_train, stops):
    raw = RAW[tid]
    if name == "detour_minor": return inject_detour(raw, 300., rng)
    if name == "detour_major": return inject_detour(raw, 750., rng)
    if name == "detour_critical": return inject_detour(raw, 1200., rng)
    if name == "smooth_detour_500": return inj_smooth_detour(raw, rng)
    if name == "stop_90s": return inject_stop(raw, FR[tid], 9, rng, c)
    if name == "stop_empirical": return inject_stop(raw, FR[tid], int(np.clip(round(3 * rng.choice(stops) / 10), 3, 24)), rng, c)
    if name == "speed_up": return inject_speed(raw, 1.5, rng)
    if name == "crawl": return inject_speed(raw, 0.5, rng)
    if name in ("speed_1.25", "speed_1.8"): return inject_speed(raw, float(name.split("_")[1]), rng)
    if name in ("crawl_0.7", "crawl_0.3"): return inject_speed(raw, float(name.split("_")[1]), rng)
    if name == "speed_jitter": return inj_speed_profile(raw, np.exp(rng.randn(8) * 0.5), rng)
    if name == "brake_pulse": return inj_speed_profile(raw, np.array([1, 0.2, 0.2, 0.2, 1, 1, 1, 1.0]), rng)
    if name == "loop": return inj_loop(raw, rng)
    if name == "stop_and_go": return inj_stop_and_go(raw, rng)
    if name == "reversal": return inj_reversal(raw, rng)
    return inj_zigzag(raw, rng)

_uid = [30000]
def build_set(ids, names, c, F_train, seed):
    rng = np.random.RandomState(seed); stops = F_train[F_train.dwell > 0].groupby("trip").dwell.max().values; stops = stops[stops > 0] if len(stops) else np.array([30.0])
    items = []
    for tid in ids:
        for name in names:
            res = make_inj(name, tid, rng, c, F_train, stops)
            if res is None: continue
            r2, mask, ws = res; _uid[0] += 1; items.append((name, mask, ws, make_frame(_uid[0], r2["t"], r2["x"], r2["y"], r2["key"])))
    return items

def batch(items, c):
    """apply context once to many injected trips; returns frame + per-item slices"""
    Fb = apply_ctx(pd.concat([it[3] for it in items], ignore_index=True), c); lens = [len(it[3]) for it in items]; off = np.r_[0, np.cumsum(lens)]
    return Fb, [(int(off[i]), int(off[i + 1])) for i in range(len(items))]

DETS = ["forest", "forest4", "rules", "lof", "ocsvm", "maxz", "asd", "hybrid"]
DLAB = {"forest": "Isolation Forest (v3)", "forest4": "Isolation Forest + speed features", "rules": "Rules only (route bands + dwell)", "lof": "Local Outlier Factor",
        "ocsvm": "One-Class SVM", "maxz": "Max z-score", "asd": "Supervised detector (trained on planted events)", "hybrid": "Hybrid: forest or supervised"}
def run_len2_flags(flag):
    f = np.asarray(flag, bool); out = np.zeros(len(f), bool)
    i = 0
    while i < len(f):
        if f[i]:
            j = i
            while j + 1 < len(f) and f[j + 1]: j += 1
            if j - i + 1 >= 2: out[i:j + 1] = True
            i = j + 1
        else: i += 1
    return out
def episodes_deb(flag):
    f = run_len2_flags(flag); return int(((f[1:] & ~f[:-1]).sum()) + (1 if len(f) and f[0] else 0))
def episodes(flag):
    f = np.asarray(flag, bool); return int(((f[1:] & ~f[:-1]).sum()) + (1 if len(f) and f[0] else 0))
def alert_levels(Fi, taus):
    ifl = np.digitize(Fi.if3.values, taus); bl = np.maximum(Fi.band_hist.values, Fi.band_poly.values); dl = np.digitize(Fi.excess_dwell.values, DWELL_LV)
    return dict(forest=ifl, rules=np.maximum(bl, dl), combined=np.maximum.reduce([ifl, bl, dl]))

# ============================================================== run the folds
FOLDS = [(order[:k - 6], order[k - 6:k], order[k:k + 6]) for k in (24, 30, 36, 42, 48, 54)]
BUDGETS = [0.25, 0.5, 1.0, 2.0]
neg_pool = {d: [] for d in DETS}; pos_pool = {k: {d: [] for d in DETS} for k in EKINDS}
ev_hit = {k: {lv: {s: [] for s in ("forest", "rules", "combined")} for lv in (1, 2, 3)} for k in EKINDS}
clean_ep = {lv: {s: [] for s in ("forest", "rules", "combined")} for lv in (1, 2, 3)}; clean_fix = {lv: {s: [] for s in ("forest", "rules", "combined")} for lv in (1, 2, 3)}
bud_ep = {d: {b: [] for b in BUDGETS} for d in DETS}; bud_hit = {d: {b: {k: [] for k in EKINDS} for b in BUDGETS} for d in DETS}
clean_hours = []; clean_trips = 0; fold_fore = []; oof = []; asd_n = []
for fi, (trn, van, ten) in enumerate(FOLDS):
    o = run_fold(trn, van, ten, mode="time", injection=False)
    D, tr, va, te, c, taus = o["D"], o["tr"], o["va"], o["te"], o["c"], o["taus"]
    fold_fore.append(dict(fore=o["fore"], base=o["base"])); T = D[te].copy(); T["fold"] = fi; T["order"] = T.trip.map({t_: dep[t_].value for t_ in ten}); oof.append(T)
    F_train = pd.concat([FR[i] for i in trn], ignore_index=True)
    if4m = IsolationForest(n_estimators=300, max_samples=256, random_state=SEED).fit(D.loc[tr, V4]); D["if4"] = -if4m.score_samples(D[V4])
    # --- supervised detector trained only on TRAIN_KINDS injected into TRAINING trips
    tr_items = build_set(trn, TRAIN_KINDS, c, F_train, SEED + 100 + fi); Fb, sl = batch(tr_items, c)
    ypos = np.zeros(len(Fb), int)
    for (a, b_), it in zip(sl, tr_items): ypos[a:b_] = it[1].astype(int)
    Xp = Fb.loc[ypos == 1, F_ASD]; Xn = D.loc[tr, F_ASD]; Xa = pd.concat([Xp, Xn]); ya = np.r_[np.ones(len(Xp)), np.zeros(len(Xn))]
    asd = xgb.XGBClassifier(n_estimators=250, learning_rate=0.05, max_depth=3, subsample=0.8, colsample_bytree=0.8, min_child_weight=3, reg_lambda=2.0, random_state=SEED, n_jobs=2).fit(Xa, ya); asd_n.append(int(len(Xp)))
    # --- baselines fitted on training fixes
    Xtr = D.loc[tr, V3].values; sc = StandardScaler().fit(Xtr); rs = np.random.RandomState(fi); sub = rs.choice(len(Xtr), min(1500, len(Xtr)), replace=False)
    lof = LocalOutlierFactor(n_neighbors=35, novelty=True).fit(sc.transform(Xtr)); svm = OneClassSVM(nu=0.1, gamma="scale").fit(sc.transform(Xtr)[sub])
    med = np.median(Xtr, 0); mad_ = np.median(np.abs(Xtr - med), 0) * 1.4826 + 1e-6
    def raw_scores(F):
        X = F[V3].values; Z = sc.transform(X)
        return dict(forest=F.if3.values, forest4=F.if4.values, lof=-lof.score_samples(Z), ocsvm=-svm.decision_function(Z), maxz=np.abs((X - med) / mad_).max(1),
                    rules=np.maximum(np.minimum(np.maximum(F.d_hist.values, F.d.values) / 1000.0, 1.0), np.minimum(F.excess_dwell.values / 180.0, 1.0)), asd=asd.predict_proba(F[F_ASD])[:, 1])
    Rv = raw_scores(D[va]); ref = {d: ecdf(Rv[d]) for d in Rv if d != "rules"}
    def scores(F):
        r = raw_scores(F); s = {d: ref[d](r[d]) for d in r if d != "rules"}; s["rules"] = r["rules"]; s["hybrid"] = np.maximum(s["forest"], s["asd"]); return s
    sval = scores(D[va]); stest = scores(D[te])
    for d in DETS: neg_pool[d].append(stest[d])
    # --- clean alert statistics at fixed concern levels
    clean = D[te]
    for tid, g in clean.groupby("trip"):
        lv = alert_levels(g, taus); clean_trips += 1; clean_hours.append(len(g) * 10 / 3600.0)
        for L in (1, 2, 3):
            for s_ in ("forest", "rules", "combined"):
                fl = lv[s_] >= L; clean_ep[L][s_].append(episodes(fl)); clean_fix[L][s_].append(float(fl.mean()))
    # --- held-out test events (all kinds), scored in batch
    te_items = build_set(ten, EKINDS, c, F_train, SEED + fi); Ft, sl2 = batch(te_items, c)
    Ft["if3"] = -o["if3"].score_samples(Ft[V3]); Ft["if4"] = -if4m.score_samples(Ft[V4]); Ft["lvl_now"], _ = levels(Ft, taus); S = scores(Ft); LV = alert_levels(Ft, taus)
    # budgeted policy: threshold on validation clean trips so that debounced false episodes <= budget, then measured on test clean trips and events
    vt = D[va].groupby("trip").indices; tt = clean.groupby("trip").indices
    for d in DETS:
        allv = sval[d]; grid = np.unique(np.quantile(allv, np.linspace(0.5, 1.0, 400)))
        for b in BUDGETS:
            tau = grid[-1]
            for g_ in grid:
                if np.mean([episodes_deb(allv[ix] > g_) for ix in vt.values()]) <= b: tau = g_; break
            bud_ep[d][b].extend([episodes_deb(stest[d][ix] > tau) for ix in tt.values()])
            for (a, b_), it in zip(sl2, te_items):
                fl = run_len2_flags(S[d][a:b_] > tau); bud_hit[d][b][it[0]].append(float((fl & it[1]).any()))
    for (a, b_), it in zip(sl2, te_items):
        kind, mask = it[0], it[1]
        for d in DETS: pos_pool[kind][d].append(S[d][a:b_][mask])
        for L in (1, 2, 3):
            for s_ in ("forest", "rules", "combined"): ev_hit[kind][L][s_].append(float((LV[s_][a:b_][mask] >= L).any()))
    print(f"fold {fi + 1}/{len(FOLDS)} done  {(_time.time() - T0) / 60:.1f} min", flush=True)

OOF = pd.concat(oof, ignore_index=True); neg = {d: np.concatenate(neg_pool[d]) for d in DETS}
det_tab = {}
for k in EKINDS:
    det_tab[k] = {}
    for d in DETS:
        pos = np.concatenate(pos_pool[k][d]); y = np.r_[np.zeros(len(neg[d])), np.ones(len(pos))]
        det_tab[k][d] = dict(auc=safe_roc(y, np.r_[neg[d], pos]), tpr5=tpr_at_fpr(neg[d], pos), n=int(len(pos)))
grp = {g: {d: float(np.nanmean([det_tab[k][d]["auc"] for k in EKINDS if KIND_GROUP[k] == g])) for d in DETS} for g in GROUPS_ORDER}
tot_hours = sum(clean_hours)
alert_tab = {L: {s: dict(eps_per_trip=float(np.sum(clean_ep[L][s]) / clean_trips), eps_per_hour=float(np.sum(clean_ep[L][s]) / tot_hours), fix_share=float(np.mean(clean_fix[L][s])),
                         recall={k: float(np.mean(ev_hit[k][L][s])) for k in EKINDS}) for s in ("forest", "rules", "combined")} for L in (1, 2, 3)}
bud_tab = {d: {b: dict(eps_per_trip=float(np.mean(bud_ep[d][b])), recall={k: float(np.mean(bud_hit[d][b][k])) for k in EKINDS},
                       group_recall={g: float(np.mean([np.mean(bud_hit[d][b][k]) for k in EKINDS if KIND_GROUP[k] == g])) for g in GROUPS_ORDER}) for b in BUDGETS} for d in DETS}

# ============================================================== forecast, ordinal, calibration, conformal
def fsum(t, g, key=0):
    v = [f["fore"][t][g][key] for f in fold_fore]; return float(np.nanmean(v)), float(np.nanstd(v))
fore = {t: {g: dict(prauc=fsum(t, g, 0)[0], sd=fsum(t, g, 0)[1], roc=fsum(t, g, 1)[0]) for g in fold_fore[0]["fore"][t]} for t in fold_fore[0]["fore"]}
base_rate = float(np.mean([f["base"]["Minor-or-worse in next 60 s"] for f in fold_fore]))
yb = (OOF.lvl_fwd >= 1).astype(int).values; praw = OOF.p_ens.values; use = OOF.fold.values >= 1
logit = lambda p: np.log(np.clip(p, 1e-4, 1 - 1e-4) / (1 - np.clip(p, 1e-4, 1 - 1e-4))).reshape(-1, 1)
pcal = praw.copy()
for f in range(1, len(FOLDS)):
    prev = OOF.fold.values < f; cur = OOF.fold.values == f
    pcal[cur] = LogisticRegression(C=1e3).fit(logit(praw[prev]), yb[prev]).predict_proba(logit(praw[cur]))[:, 1]
OOF["p_cal"] = pcal
def ece(y, p, bins=8):
    q = np.quantile(p, np.linspace(0, 1, bins + 1)); q[-1] += 1e-9; e = 0
    for lo, hi in zip(q[:-1], q[1:]):
        m = (p >= lo) & (p < hi)
        if m.sum(): e += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(e)
cal = dict(raw_ece=ece(yb[use], praw[use]), cal_ece=ece(yb[use], np.clip(pcal[use], 0, 1)), raw_brier=float(brier_score_loss(yb[use], praw[use])), cal_brier=float(brier_score_loss(yb[use], np.clip(pcal[use], 0, 1))))
conf = {}
for al in (0.1, 0.2, 0.3):
    cv, sg_, sa = [], [], []
    for f in range(1, len(FOLDS)):
        prev = OOF.fold.values < f; cur = OOF.fold.values == f
        pc_ = np.clip(pcal[prev], 1e-4, 1 - 1e-4); scs = np.where(yb[prev] == 1, 1 - pc_, pc_); n_ = len(scs); q = np.quantile(scs, min(1.0, math.ceil((n_ + 1) * (1 - al)) / n_))
        p_ = np.clip(pcal[cur], 1e-4, 1 - 1e-4); y_ = yb[cur]; a1, a0 = (1 - p_) <= q, p_ <= q; cov = np.where(y_ == 1, a1, a0); sz = a1.astype(int) + a0.astype(int); si = sz == 1
        cv.append(cov.mean()); sg_.append(si.mean()); sa.append(cov[si].mean() if si.any() else np.nan)
    conf[al] = dict(target=1 - al, coverage=float(np.mean(cv)), singleton=float(np.mean(sg_)), singleton_acc=float(np.nanmean(sa)))
fop = []
for th in (0.5, 0.7, 0.85):
    A = OOF[use]; fl = A.p_cal.values >= th; y = yb[use]
    fop.append(dict(th=th, precision=float(y[fl].mean()) if fl.any() else np.nan, recall=float(fl[y == 1].mean()), eps_per_trip=sum(episodes(g.p_cal.values >= th) for _, g in A.groupby("trip")) / A.trip.nunique(), share=float(fl.mean())))
Y = OOF.lvl_fwd.values; pred = OOF.pred_lvl.values; now = OOF.lvl_now.values
om = lambda p_: dict(qwk=float(cohen_kappa_score(Y, p_, weights="quadratic")), mae=float(np.abs(Y - p_).mean()), within1=float((np.abs(Y - p_) <= 1).mean()))
ordm = dict(model=om(pred), persistence=om(now))

# ============================================================== 5) real labels (optional)
LAB = None
if ARGS.labels:
    L = pd.read_csv(ARGS.labels)
    fmap = {r["file"]: i for i, r in enumerate(recs)}; imap = {r.get("tripId"): i for i, r in enumerate(recs)}
    L["idx"] = [fmap.get(f) if isinstance(f, str) and f in fmap else imap.get(t_) for f, t_ in zip(L["file"] if "file" in L else [None] * len(L), L["trip_id"] if "trip_id" in L else [None] * len(L))]
    unmatched = int(L.idx.isna().sum()); L = L.dropna(subset=["idx"]); L = L[L.incident.notna()]; L["idx"] = L.idx.astype(int)
    Ls = L[L.idx.isin(set(OOF.trip.unique()))]; rows = []
    for _, r in Ls.iterrows():
        g = OOF[OOF.trip == r.idx]; rows.append(dict(y=int(r.incident), max_risk=float(g.p_cal.max()), max_level=float(g.lvl_now.max()), concern_share=float((g.lvl_now >= 1).mean())))
    R = pd.DataFrame(rows); LAB = dict(n_rows=int(len(L)), unmatched=unmatched, n_scored=int(len(R)), n_pos=int(R.y.sum()) if len(R) else 0, not_scored=int(len(L) - len(R)))
    if len(R) and 0 < R.y.sum() < len(R):
        LAB.update(trip_auc_maxrisk=safe_roc(R.y.values, R.max_risk.values), trip_auc_maxlevel=safe_roc(R.y.values, R.max_level.values), trip_auc_share=safe_roc(R.y.values, R.concern_share.values),
                   trip_ap_maxrisk=safe_pr(R.y.values, R.max_risk.values), trip_base=float(R.y.mean()))
    if "incident_time" in L.columns and L.incident_time.notna().any():
        yfix = np.zeros(len(OOF), int); lab_t = {r.idx: pd.to_datetime(r.incident_time, utc=True) for _, r in L.iterrows() if isinstance(r.incident_time, str)}
        for tid, tt_ in lab_t.items():
            m = (OOF.trip == tid).values
            if m.any(): dt_ = (tt_ - OOF.t[m]).dt.total_seconds(); yfix[np.where(m)[0][((dt_ > 0) & (dt_ <= 60)).values]] = 1
        mk = OOF.trip.isin(list(lab_t.keys())).values
        if yfix[mk].sum() > 0: LAB.update(fix_pr=safe_pr(yfix[mk], OOF.p_cal.values[mk]), fix_roc=safe_roc(yfix[mk], OOF.p_cal.values[mk]), fix_base=float(yfix[mk].mean()))

# ============================================================== gates (fixed before looking at results)
G = []
def gate(name, crit, val, ok): G.append(dict(name=name, crit=crit, val=val, status="N/A" if ok is None else ("PASS" if bool(ok) else "FAIL")))
mn = "v3 + current concern level"; per = "Persistence (concern now → concern soon)"; mt = "Minor-or-worse in next 60 s"
mean_auc = lambda d: float(np.mean([det_tab[k][d]["auc"] for k in EKINDS]))
gate("Data quality", "no critical issues in the CSVs", f"{DQ['n_issue_trips']} of {DQ['n_trips']} trips flagged", not DQ["critical"])
gate("Forecast beats persistence", "PR-AUC gain larger than one fold SD", f"{fore[mt][mn]['prauc']:.2f} vs {fore[mt][per]['prauc']:.2f} (SD {fore[mt][mn]['sd']:.2f})", fore[mt][mn]["prauc"] - fore[mt][per]["prauc"] > fore[mt][mn]["sd"])
gate("Detects rule-designed events", "forest mean AUC ≥ 0.90", f"{grp['designed']['forest']:.2f}", grp["designed"]["forest"] >= 0.90)
gate("Detects events the rules miss", "forest mean AUC ≥ 0.80 and above rules", f"forest {grp['unseen']['forest']:.2f} vs rules {grp['unseen']['rules']:.2f}", grp["unseen"]["forest"] >= 0.80 and grp["unseen"]["forest"] > grp["unseen"]["rules"])
gate("Beats generic detectors", "forest mean AUC (all events) ≥ best of LOF, SVM, max-z", " / ".join(f"{d} {mean_auc(d):.2f}" for d in ("forest", "lof", "ocsvm", "maxz")), mean_auc("forest") >= max(mean_auc(d) for d in ("lof", "ocsvm", "maxz")))
best_sp = max(("forest", "forest4", "lof", "asd", "hybrid"), key=lambda d: grp["speed_heldout"][d])
gate("Speed changes (held-out magnitudes)", "best detector mean AUC ≥ 0.80", f"best: {DLAB[best_sp]} {grp['speed_heldout'][best_sp]:.2f} (forest {grp['speed_heldout']['forest']:.2f})", grp["speed_heldout"][best_sp] >= 0.80)
gate("Supervised detector generalises", "AUC on never-seen event types ≥ 0.80 and ≥ forest", f"asd {grp['unseen']['asd']:.2f} vs forest {grp['unseen']['forest']:.2f}", grp["unseen"]["asd"] >= 0.80 and grp["unseen"]["asd"] >= grp["unseen"]["forest"])
gate("Speed events of a new kind", "hybrid mean AUC ≥ 0.75 on jitter and brake pulse", f"hybrid {grp['speed_newtype']['hybrid']:.2f}, supervised {grp['speed_newtype']['asd']:.2f}, forest {grp['speed_newtype']['forest']:.2f}", grp["speed_newtype"]["hybrid"] >= 0.75)
al = alert_tab[ARGS.alert_level]["combined"]
gate(f"False alarms, fixed level ≥ {LEVELS[ARGS.alert_level]}", f"≤ {ARGS.max_false_alerts:g} episodes per clean trip", f"{al['eps_per_trip']:.2f} per trip ({al['eps_per_hour']:.0f} per hour)", al["eps_per_trip"] <= ARGS.max_false_alerts)
bb = min(BUDGETS, key=lambda x: abs(x - ARGS.budget)); bf = bud_tab["forest"][bb]["eps_per_trip"]
gate("False-alarm budget respected", f"measured on test trips ≤ 1.25 × budget ({bb:g} per trip), forest", f"{bf:.2f} per trip", bf <= 1.25 * bb)
gate("Calibration", "ECE after calibration ≤ 0.08", f"{cal['cal_ece']:.3f} (raw {cal['raw_ece']:.3f})", cal["cal_ece"] <= 0.08)
gate("Conformal coverage", "actual within ±0.05 of target at 90%", f"{conf[0.1]['coverage']:.2f} vs 0.90", abs(conf[0.1]["coverage"] - 0.9) <= 0.05)
if LAB and "trip_auc_maxrisk" in LAB:
    enough = LAB["n_scored"] >= 30 and LAB["n_pos"] >= 5
    gate("Real labels: trip-level accuracy", "AUC ≥ 0.70 with ≥ 30 labelled trips and ≥ 5 incidents", f"AUC {LAB['trip_auc_maxrisk']:.2f}, {LAB['n_scored']} trips, {LAB['n_pos']} incidents", (LAB["trip_auc_maxrisk"] >= 0.7) if enough else None)
else: gate("Real labels: trip-level accuracy", "AUC ≥ 0.70 with ≥ 30 labelled trips and ≥ 5 incidents", "no usable labels provided", None)

# ============================================================== report
f2 = lambda v: "–" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.2f}"; pc = lambda v: "–" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v * 100:.0f}%"
def cell(v, st=""): return (v, st)
def _td(i, v, st): return "<td" + (" class='n'" if i else "") + ((" style='" + st + "'") if st else "") + ">" + str(v) + "</td>"
def tbl(head, rows): return "<div class='tw'><table><tr>" + "".join("<th>" + c + "</th>" for c in head) + "</tr>" + "".join("<tr>" + "".join(_td(i, v, st) for i, (v, st) in enumerate(r)) + "</tr>" for r in rows) + "</table></div>"
def heat(v, lo=0.5):
    if v is None or np.isnan(v): return ""
    return f"background:color-mix(in srgb, var(--accent) {int(max(0, (v - lo) / (1 - lo)) * 60)}%, transparent)"
sec = []
gate_rows = [[cell(g["name"]), cell(g["crit"]), cell(g["val"]), cell("<b class='" + g["status"].lower().replace("/", "") + "'>" + g["status"] + "</b>")] for g in G]
npass = sum(g["status"] == "PASS" for g in G); nfail = sum(g["status"] == "FAIL" for g in G); nna = sum(g["status"] == "N/A" for g in G)
sec.append(f"<section><h2>Verdict</h2><p><b>{npass} passed, {nfail} failed, {nna} not applicable.</b> Gates were fixed before the numbers were seen; change them in <code>evaluate.py</code> if your adviser requires different ones.</p>{tbl(['Check', 'Criterion', 'Result', 'Status'], gate_rows)}</section>")
sec.append("<section><h2>0 · Data quality</h2><p>" + f"{DQ['n_trips']} trips, {DQ['n_fixes']:,} fixes, median interval {DQ['median_dt']:.0f} s, largest gap {DQ['max_gap']:.0f} s. " + (f"{DQ['n_issue_trips']} trips flagged:" if DQ["issues"] else "No issues found.") + "</p>" +
           ("<ul>" + "".join(f"<li><code>{html_mod.escape(f)}</code>: {k}</li>" for f, k in DQ["issues"][:30]) + "</ul>" if DQ["issues"] else "") + "</section>")
FOP_TABLE = tbl(["Alert when risk ≥", "Precision", "Recall", "Alert episodes / trip", "Share of fixes alerting"], [[cell("%.2f" % o_["th"]), cell(pc(o_["precision"])), cell(pc(o_["recall"])), cell(f2(o_["eps_per_trip"])), cell(pc(o_["share"]))] for o_ in fop])
rows = [[cell(g), cell("%.3f ± %.2f" % (v["prauc"], v["sd"])), cell("%.3f" % v["roc"])] for g, v in fore[mt].items()]
sec.append(f"<section><h2>1 · Forecast accuracy</h2><p>Will the next 60 s contain a Minor-or-worse moment? Six rolling-origin folds; each fold trains on earlier trips and tests on the next six. Base rate {base_rate:.2f}. Ordinal level quadratic kappa {ordm['model']['qwk']:.2f} vs persistence {ordm['persistence']['qwk']:.2f}; mean level error {ordm['model']['mae']:.2f} vs {ordm['persistence']['mae']:.2f}.</p>{tbl(['Model', 'PR-AUC (mean ± SD)', 'ROC-AUC'], rows)}<h3>Operating points (calibrated risk)</h3>{FOP_TABLE}</section>")
head2 = ["Planted event"] + [DLAB[d] for d in DETS]
def drows(key, lo):
    out = []
    for g in GROUPS_ORDER:
        out.append([cell("<b>" + GTITLE[g] + "</b>", "background:var(--accent-soft)")] + [cell("", "background:var(--accent-soft)")] * len(DETS))
        for k in EKINDS:
            if KIND_GROUP[k] == g: out.append([cell(KLAB[k])] + [cell(f2(det_tab[k][d][key]), heat(det_tab[k][d][key], lo)) for d in DETS])
        if key == "auc": out.append([cell("<i>group mean</i>")] + [cell(f2(grp[g][d]), heat(grp[g][d], lo)) for d in DETS])
    return out
sec.append("<section><h2>2 · Detection of planted events</h2><p>Known abnormal events are inserted into held-out trips. AUC against clean fixes (0.50 = chance). The <b>supervised detector</b> is trained only on detours, a 90 s stop and speed ×1.5 / ×0.5 planted in <i>training</i> trips; everything in the last two groups is new to it. All scores are fitted on earlier trips only. <b>Caveat:</b> planted events are synthetic, so a detector trained on them may not transfer to real behaviour.</p>" + tbl(head2, drows("auc", 0.5)) + "<h3>Detected at a fixed 5% false-alarm rate</h3>" + tbl(head2, drows("tpr5", 0.0)) + "</section>")
a_rows = []
for L_ in (1, 2, 3):
    for s_, nm in (("forest", "Forest"), ("rules", "Rules"), ("combined", "Combined (model)")):
        a = alert_tab[L_][s_]; rec = a["recall"]; gm = lambda g: np.mean([rec[k] for k in EKINDS if KIND_GROUP[k] == g])
        a_rows.append([cell(f"≥ {LEVELS[L_]}"), cell(nm), cell(f2(a["eps_per_trip"])), cell(f"{a['eps_per_hour']:.0f}"), cell(pc(a["fix_share"])), cell(pc(gm("designed"))), cell(pc(gm("unseen"))), cell(pc((gm("speed_seen") + gm("speed_heldout") + gm("speed_newtype")) / 3))])
b_rows = []
for d in ("forest", "forest4", "rules", "lof", "asd", "hybrid"):
    for b in BUDGETS:
        t_ = bud_tab[d][b]; gr = t_["group_recall"]
        b_rows.append([cell(DLAB[d] if b == BUDGETS[0] else ""), cell(f"{b:g}"), cell(f2(t_["eps_per_trip"])), cell(pc(gr["designed"])), cell(pc(gr["speed_seen"])), cell(pc(gr["speed_heldout"])), cell(pc(gr["speed_newtype"])), cell(pc(gr["unseen"]))])
sec.append("<section><h2>3 · False alarms</h2><p>An alert “episode” is a run of consecutive fixes at or above a level. False alarms are measured on clean held-out trips, so they include normal stops and curves.</p><h3>Fixed concern levels (v3 policy)</h3>" +
           tbl(["Alert level", "Source", "False episodes / clean trip", "per hour", "Share of clean fixes alerting", "Recall: designed-for", "Recall: not designed for", "Recall: speed changes"], a_rows) +
           "<h3>Budgeted policy (new)</h3><p>For each detector the alert threshold is chosen on the <i>validation</i> trips to meet a false-alarm budget (episodes per clean trip), and an alert needs at least two consecutive fixes above the threshold. The “measured” column is the false-alarm rate actually seen on separate test trips; the recall columns show what each budget buys.</p>" +
           tbl(["Detector", "Budget (episodes / trip)", "Measured on test trips", "Recall: designed-for", "Recall: speed, trained magnitudes", "Recall: speed, held-out magnitudes", "Recall: speed, new kinds", "Recall: other unseen types"], b_rows) + "</section>")
sec.append(f"<section><h2>4 · Calibration and uncertainty</h2><p>Calibration error (ECE) {cal['raw_ece']:.3f} → {cal['cal_ece']:.3f} after Platt scaling fitted on earlier folds; Brier {cal['raw_brier']:.3f} → {cal['cal_brier']:.3f}.</p>" +
           tbl(["Conformal target", "Actual coverage", "Moments with a single answer", "Accuracy when single"], [[cell(pc(v['target'])), cell(pc(v['coverage'])), cell(pc(v['singleton'])), cell(pc(v['singleton_acc']))] for v in conf.values()]) + "</section>")
if LAB:
    lt = [[cell("Rows in labels file"), cell(str(LAB["n_rows"]))], [cell("Could not be matched to a trip"), cell(str(LAB["unmatched"]))], [cell("Scored out-of-sample (trips after the first 24)"), cell(str(LAB["n_scored"]))], [cell("Incidents among scored trips"), cell(str(LAB["n_pos"]))]]
    for k_, nm in (("trip_auc_maxrisk", "Trip-level AUC, highest risk in trip"), ("trip_auc_maxlevel", "Trip-level AUC, highest concern level"), ("trip_auc_share", "Trip-level AUC, share of fixes in concern"), ("trip_ap_maxrisk", "Trip-level average precision"), ("fix_roc", "Fix-level AUC, incident within next 60 s"), ("fix_pr", "Fix-level PR-AUC")):
        if k_ in LAB: lt.append([cell(nm), cell(f2(LAB[k_]))])
    warn = "" if LAB["n_scored"] >= 30 and LAB["n_pos"] >= 5 else "<p><b>Too few labels for a reliable estimate.</b> Aim for at least 30 labelled trips and 5 incidents; treat these numbers as a smoke test.</p>"
    sec.append("<section><h2>5 · Real labels</h2>" + warn + tbl(["Measure", "Value"], lt) + "</section>")
else: sec.append("<section><h2>5 · Real labels</h2><p>No labels file was provided, so nothing here measures real safety. Re-run with <code>--labels labels.csv</code> (columns <code>file</code> or <code>trip_id</code>, <code>incident</code> 0/1, optional <code>incident_time</code>).</p></section>")
CSS = """:root{--bg:#fbfaf7;--panel:#fff;--fg:#1d2327;--muted:#5d686f;--line:#e3e0d8;--accent:#0f6b73;--accent-soft:#dcefef}@media(prefers-color-scheme:dark){:root{--bg:#14181b;--panel:#1b2125;--fg:#e8ecee;--muted:#9aa6ad;--line:#2d363b;--accent:#58c1c9;--accent-soft:#17363a}}
body{margin:0;background:var(--bg);color:var(--fg);font:16px/1.6 system-ui,sans-serif}.w{max-width:1100px;margin:0 auto;padding:0 20px 60px}h1{font-size:34px;margin:40px 0 4px}h2{font-size:24px;margin:34px 0 8px}h3{font-size:17px;margin:22px 0 6px}.tw{overflow-x:auto;background:var(--panel);border:1px solid var(--line);border-radius:10px}
table{width:100%;border-collapse:collapse;font-size:13.5px}th,td{text-align:left;padding:7px 9px;border-bottom:1px solid var(--line);vertical-align:top}th{color:var(--muted);font-size:11.5px;text-transform:uppercase;letter-spacing:.04em}td.n{text-align:right;font-variant-numeric:tabular-nums}
code{background:var(--accent-soft);padding:1px 6px;border-radius:4px;font-size:.88em}b.pass{color:#2f8f6b}b.fail{color:#c4412b}b.na{color:var(--muted)}.sub{color:var(--muted)}"""
page = ("<!doctype html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>LIGTAS Evaluation Report</title><style>" + CSS + "</style></head><body><div class='w'><h1>LIGTAS evaluation report</h1><p class='sub'>Dataset: <code>"
        + html_mod.escape(ARGS.dataset) + "</code> · " + str(len(recs)) + " trips · generated " + pd.Timestamp.now().strftime('%Y-%m-%d %H:%M') + " · all detectors and models are fitted on earlier trips only</p>" + "".join(sec) + "</div></body></html>")
open(os.path.join(ARGS.out_dir, "evaluation_report.html"), "w").write(page)
json.dump(dict(gates=G, data=DQ, fore=fore, detect=det_tab, groups=grp, alerts={str(k): v for k, v in alert_tab.items()}, budgets={d: {str(b): v for b, v in t.items()} for d, t in bud_tab.items()},
               cal=cal, conf={str(k): v for k, v in conf.items()}, forecast_ops=fop, labels=LAB, dlab=DLAB, klab=KLAB, kind_group=KIND_GROUP, gtitle=GTITLE, asd_train_positives=asd_n), open(os.path.join(ARGS.out_dir, "evaluation_report.json"), "w"), indent=1, default=float)
print("done in", round((_time.time() - T0) / 60, 1), "min;", npass, "pass", nfail, "fail", nna, "n/a")
for g in G: print(g["status"], "|", g["name"], "|", g["val"])
print("GROUPS"); [print(" ", g, {d: round(v, 2) for d, v in dd.items()}) for g, dd in grp.items()]
