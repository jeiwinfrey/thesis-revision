# LIGTAS-CASCADE: how the algorithm works, and what justifies each part

**Working title:** *Utilizing Isolation Forest and XGBoost Algorithms in Forecasting Anomalous Trip Behavior as a Commuter Safety Indicator*

> **Status.** Prototype run on 60 synthetic trips (one route, one vehicle type, daytime, 10-second GPS). There are no incident or rider-rating labels. "Risk" here means *how unusual the next 60 seconds are likely to be*. It is a proxy for safety, not a measurement of danger. Numbers below come from `pipeline3.py` and `evaluate.py` and will change when more data arrives.
>
> **Reference honesty.** Every reference in section 9 was found by web search in this project. Only Edwards & Hosein (2025) was read in full. All others were checked at abstract or search-summary level. Confirm authors, years, pages and claims against the originals before citing in the manuscript.

---

## 1. What the model is for

Given only **timestamp, latitude and longitude** for a vehicle on a known route, produce, every 10 seconds:

1. a **concern level** for the current moment (Normal, Minor, Major, Critical);
2. a **risk** (0 to 1) that the next 60 seconds will contain a Minor-or-worse moment, shown as a **Safety Index** (`100 × (1 − risk)`);
3. a **forecast level** for the worst moment expected in the next 60 seconds;
4. **how sure** the model is (a risk interval and a confidence set);
5. **reasons** for the score.

In a proof-of-concept app the output would drive a prompt to the rider ("are you okay?") or an alert to a chosen contact, in the manner of Edwards & Hosein [9].

## 2. Input data

Each trip is a CSV with `timestamp, latitude, longitude`. The route geometry (`project.json`) is used only to measure position along the route and distance from it. Everything else is derived. No labels are used by the core model.

## 3. Pipeline at a glance

```
GPS fixes (time, lat, lon)
   │
   ▼  Stage 0  features
   speed, acceleration, jerk, turn rate, dwell, rolling stats,
   position along route (s), distance from route line (d), heading deviation,
   distance to earlier trips (d_hist), run-up features (speed trend, rate of change of d)
   │
   ▼  Stage 1  context-conditioned baseline + Isolation Forest   ("teacher")
   "unusual for THIS place, direction and hour block"  →  unusualness score
   │
   ▼  Concern level now  =  worst of { forest score, route-deviation band, excess dwell }
   │
   ▼  Stage 2  XGBoost forecast   ("student")
   P(level ≥ Minor / Major / Critical within the next 60 s)   →  ordinal forecast
   │
   ▼  Stage 3  calibration + uncertainty
   Platt scaling, 5-model bootstrap spread, conformal prediction sets
   │
   ▼  Stage 4  detection add-on   (v4)
   supervised detector trained on planted events  →  hybrid with the forest
   │
   ▼  Stage 5  alert policy
   threshold chosen on validation trips to meet a false-alarm budget, 2 consecutive fixes required
```

## 4. Stage by stage

For each stage: **what it does**, **parameters**, **justification**. Justification is graded:

- **Supported**: a source reports the same method for a similar purpose.
- **Partly**: a source supports the idea, but not this exact setting or value.
- **Design choice**: no source found; chosen by the project and tested only on this data.

### Stage 0. Features from raw GPS

Per 10-second fix: speed `v`, acceleration `a`, jerk, turn rate, seconds stationary (`dwell`), 60-second rolling mean, standard deviation, minimum acceleration and stop count, position along the route `s`, distance from the route polyline `d`, heading deviation from the road direction, local hour and day of week. Run-up features (v3): speed trend over 30 s, time since last stop, distance to the next usual stop zone, rate of change of `d`. Speed features (v4): ratio of recent to earlier moving speed, moving-only speed z-score, acceleration sign changes, oscillation of `d`, reversal count. All features look backward only.

**Justification: Partly.** GPS-derived hard acceleration, hard deceleration and long stops correlate with pedestrian and bicycle crashes near bus stops [7]; deceleration and jerk from smartphone speed correlate with culpable crashes among bus rapid transit drivers [8]. Both used faster or richer data than 10-second fixes, and [28] reports that about 1 Hz GPS is needed to detect hard braking, so **jerk and braking features are weak at 10 s**. Time of day as a feature is motivated by survey evidence that perceived safety varies by hour [25, 26] (not Philippine data).

### Stage 1. Context-conditioned baseline and Isolation Forest

**Baseline.** For each 100 m stretch × direction × hour block (night, morning peak, midday, afternoon peak), the median and robust spread of moving speed are learned from training trips only. Blocks with few trips are shrunk toward the all-day stretch baseline: `w = n / (n + 8)`. The result is a speed residual `v − baseline` and a robust z-score.

**Forest.** `IsolationForest`, 300 trees, 256 samples per tree, fitted on training trips only, scoring `−score_samples`. Inputs: speed, acceleration, jerk, turn rate, distance from route, speed residual, speed variability, dwell, heading deviation, excess dwell, distance to earlier trips, and (v3) speed z-score, run-up features. Higher score = easier to isolate = more unusual.

**Justification.**
- Isolation Forest isolates anomalies without labels, is linear in time and uses little memory: **Supported** [1]. It has been applied to driving and trajectory data without labels [21, 22], and isolation-based methods detect anomalous taxi routes well on real GPS (AUC above 0.99 reported) [10].
- Standard forests show axis-aligned score artefacts [2]. A rotated-forest variant was tested in v2 and gave mixed results, so the standard forest is kept.
- The **contextual baseline** (score deviation from the local norm, not the global cloud) is **Partly**: [10] conditions on source-destination pairs and [14] scores stop duration against the typical stay in each segment, but the specific construction here, including hour blocks and shrinkage, is **a design choice**. Tested result: the time-aware version was **not clearly better** than a morning/afternoon split or a single all-day baseline on this data (no night trips, 134 midday fixes).
- **Result vs alternatives:** the forest is about tied with Local Outlier Factor and One-Class SVM on planted events (mean AUC 0.81 vs 0.82 and 0.80). It beats fixed rules and max-z. No source claims the forest should beat those alternatives here.

### Concern level now (Normal / Minor / Major / Critical)

The worst of three signals:

| Signal | Minor | Major | Critical |
|---|---|---|---|
| Forest score (percentile of training scores) | ≥ 90th | ≥ 97.5th | ≥ 99.5th |
| Route deviation (distance to route line or to earlier trips) | > 150 m | > 500 m | > 1,000 m |
| Excess dwell (seconds stationary beyond the local 90th percentile) | ≥ 30 s | ≥ 90 s | ≥ 180 s |

**Justification.**
- Deviation bands: **Partly.** Edwards & Hosein [9] define Normal ≤ 150 m, Minor ≤ 500 m, Major ≤ 1,000 m, Critical > 1,000 m, measured as minimum distance to points from previous trips. I read the full text: the thresholds are "predefined", no empirical basis is given, and the evaluation is four simulated scenarios on one route.
- Excess dwell as an anomaly signal: **Supported** in principle. Unexpected stops are treated as safety-relevant, scored against the typical stay time per segment [13, 14], and long stops had the strongest correlation with crashes in [7]. The **30/90/180 s values are a design choice.**
- Forest percentiles (90 / 97.5 / 99.5) and "worst of three": **design choices.** Changing the 90 % cutoff changes the label's base rate; a sensitivity check (5 % to 20 %) showed lift over base rate falls as the cutoff rises, and the ranking of anomalies is stable across seeds (top-10 % overlap 0.87).

### Stage 2. Forward forecast with XGBoost

**Target:** the worst concern level in the next 6 fixes (60 s). **Model:** gradient-boosted trees (`max_depth` 3, learning rate 0.04, up to 400 trees, early stopping on validation trips, no class re-weighting). **Features:** Stage 0, deviation and dwell features, run-up features, plus the current level and forest score. **Ordinal output:** three cumulative binary models, `P(level ≥ Minor)`, `≥ Major`, `≥ Critical`, made monotone and turned into a level with thresholds tuned on validation trips.

**Justification.**
- XGBoost for tabular prediction: **Supported** [3]. On space-time-only inputs, XGBoost has been competitive for crime-risk forecasting [23, 24] (the latter a course project).
- Cumulative binary models for ordered classes: **Supported** [15].
- Using an unsupervised score to create the target for a supervised model: **Supported as a pattern** [19, 20] (healthcare billing and network anomaly detection).
- The **60-second horizon is a design choice** with no source.
- **Tested result:** the forecast beats "the level now continues" (PR-AUC 0.71 vs 0.51 across six folds), but the gain over v2 features is inside fold-to-fold spread. Ordinal forecasts agree with truth somewhat better than persistence (kappa 0.50 vs 0.43) but are worse on exact accuracy and almost never correct at the Critical level (8 of 71).

### Stage 3. Calibration and uncertainty

- **Calibration:** Platt scaling fitted on earlier folds and tested on later ones. Measured by expected calibration error (ECE) and Brier score. ECE fell from 0.106 to 0.045.
- **Spread:** five XGBoost models trained on trip-level bootstrap resamples; ± one standard deviation gives a risk interval. Tested result: **not a reliable warning** (errors were not clearly larger where models disagree).
- **Conformal prediction:** sets of "normal likely", "concern likely" or "uncertain", with calibration scores from earlier folds. At the 90 % target, actual coverage was 90 %, 61 % of moments got a single answer, and those were right 84 % of the time; coverage varied by fold. An adaptive variant retunes the miscoverage level online.

**Justification.** Boosted trees distort probabilities and calibration (Platt or isotonic) helps; Platt is safer when data is scarce: **Supported** [5]. ECE as the measure: **Supported** [18] (a neural-network paper, but the metric is general). Conformal prediction: **Supported** [16], including time-series and abstention extensions; adaptive coverage under drift: **Supported** [17]. Standard conformal guarantees assume exchangeability, which time series violate [29], so the observed fold-to-fold variation is expected. Using Platt over isotonic here is justified by [5] and by the results (isotonic had much worse log loss).

### Stage 4. Supervised detector and hybrid (v4)

**What it does.** An XGBoost classifier is trained on clean training fixes versus fixes inside events **planted in training trips only** (detours of 300, 750, 1,200 m, a 90-second stop, speed ×1.5 and ×0.5). Position and time-of-day features are excluded so it cannot memorise the route. The **hybrid** alerts if the forest *or* the supervised detector fires (maximum of their percentile scores).

**Tested result.** On speed changes of magnitudes it never saw it reaches AUC 0.86 (forest 0.65), and on two new kinds of speed event (random jitter, a brief brake pulse) 0.86. On event types it never saw (loops, backtracking, zigzag, stop-and-go) it is weaker alone (0.83 vs forest 0.85), and the hybrid reaches 0.88.

**Justification: Design choice, with indirect support only.**
- Planting detours and speed changes into trajectories to create labelled test cases is a common protocol [11, 12]. Those papers use injection for **evaluation**. Using planted events as **training data** for a supervised detector is a different step, and **I have not found or verified a source for it**. It resembles "synthetic anomaly exposure" in the anomaly-detection literature, but I did not search for or verify a citation.
- The "forest or supervised" rule is an engineering choice with no source.
- **Caveat that matters:** the planted and test events come from the same generator. Held-out magnitudes and new speed kinds are a fair test of the idea, not evidence of transfer to real riders. Evidence on synthetic-to-real transfer comes from other fields and is mixed [30].

### Stage 5. Alert policy

For each detector, the alert threshold is chosen on **validation trips** as the smallest threshold whose debounced false episodes per clean trip are at or under a budget (0.25, 0.5, 1, or 2). An alert needs **two consecutive fixes** above the threshold. The false-alarm rate is then measured on separate test trips.

**Justification: Design choice.** Choosing thresholds against a false-alarm budget is standard practice but I have no source for it. Edwards & Hosein [9] note that false alarms lead to user desensitisation, which motivates the budget, but they do not specify one. Tested result: budgets are met only roughly (0.8× to 1.8×), because six validation trips are few. At the strictest budget (about one false alarm per four clean trips) the hybrid caught 99 % of designed-for events, 75 % of held-out speed changes and 60 % of unseen types.

## 5. Justification audit

| Component | Verdict | Sources | Note |
|---|---|---|---|
| Isolation Forest for unlabelled anomalies | **Supported** | [1, 10, 21, 22] | Not better than LOF / One-Class SVM here |
| XGBoost forecaster | **Supported** | [3, 23, 24] | [24] is a course project |
| Blocked, chronological validation | **Supported** | [6] | Random splits overstate accuracy on dependent data |
| Platt calibration, ECE | **Supported** | [5, 18] | Platt chosen for small data |
| Ordinal output (cumulative binary) | **Supported** | [15] | Performance weak at Critical |
| Conformal sets, adaptive version | **Supported** | [16, 17, 29] | Coverage varies by fold |
| Unsupervised score → labels for a supervised model | **Supported** (pattern) | [19, 20] | Different domains |
| Planting events to test detectors | **Supported** | [11, 12] | For evaluation only |
| Excess dwell | **Supported** (idea) / design choice (30/90/180 s) | [7, 13, 14] | |
| Route-deviation bands 150/500/1,000 m | **Partly** | [9] | Published but "predefined"; no empirical basis |
| GPS motion features relate to safety | **Partly** | [7, 8] | Pedestrian/cyclist crashes and culpable bus crashes, with richer data than 10 s GPS |
| Hour of day as a feature | **Partly** | [25, 26] | Survey evidence, not Philippine, not trip data |
| Gendered transit risk motivates rider-profile column | **Partly** | [27] | Slide deck, method only |
| Contextual baseline (stretch × direction × hour) | **Partly** | [10, 14] | Construction is ours; no measured benefit yet |
| 60-second forecast horizon | **Design choice** | none | |
| 90th / 97.5th / 99.5th percentile event cutoffs | **Design choice** | none | Base rate depends on it |
| "Worst of three signals" for concern level | **Design choice** | none | |
| Supervised detector trained on planted events | **Design choice** | none (injection papers cover evaluation only) | Biggest unsupported step |
| Hybrid "forest or supervised" | **Design choice** | none | |
| Budgeted threshold + 2-fix debounce | **Design choice** | none | |
| Bootstrap ensemble spread | **Design choice** | none | Tested: not reliable |
| "Unusual behaviour indicates safety concern" | **Partly, indirectly** | [7, 8, 13] | Never tested on real incidents |

**Where an RRL is still needed**
1. Synthetic or "outlier exposure" training for anomaly detectors, and whether it transfers to real data.
2. Contextual (conditional) anomaly detection in trajectories.
3. Safety proxies for **passengers inside** the vehicle, not pedestrians near it.
4. Philippine commuter safety evidence (the only item found was a method-only slide deck [27]).
5. Alert-fatigue and false-alarm budgets in safety apps.

## 6. How the algorithm is verified

`evaluate.py` produces `evaluation_report.html`. It checks, with fixed pass/fail gates: CSV data quality; forecast against persistence on six rolling-origin folds (a trip is never split); detection of planted events in four groups (designed-for, speed changes with trained magnitudes, speed changes with never-seen magnitudes and kinds, and event types the rules were not designed for) against rules, LOF, One-Class SVM and max-z; false alarms; calibration; conformal coverage; and, when you supply `labels.csv`, real trip-level and fix-level accuracy. A control run with random labels scored AUC 0.53, i.e. chance, and failed, as it should.

Latest run: 9 pass, 3 fail, 1 not applicable. Failures: forest does not beat LOF; the supervised detector alone does not beat the forest on unseen types; the fixed Major-level policy gives 1.06 false episodes per clean trip (limit 1.0).

## 7. What the output can and cannot support

- **Can support:** "this trip is behaving unusually for this place and time, and here is why"; a calibrated 0–1 number for how likely unusual behaviour is in the next minute; planted detours and long stops are detected reliably.
- **Cannot support:** that flagged moments are unsafe; that the model transfers to real riders, other routes, vehicles, night or rain; that it warns ahead of sudden events (it gives only a weak hint when a problem builds gradually).

## 8. Reproduce

```bash
python3 pipeline3.py  /path/to/01-test-route results_v3.json sample_output_v3.csv
python3 evaluate.py   /path/to/01-test-route [--labels labels.csv]
python3 build.py      # rebuilds ligtas-cascade.html
```

Requires numpy, pandas, scipy, scikit-learn, xgboost. Random seed 7.

## 9. References

Status key: **Full** = read in full; **Abstract** = abstract or search summary only; **Title** = title and snippet only.

| # | Reference | Link | Status |
|---|---|---|---|
| 1 | Liu, F. T., Ting, K. M., & Zhou, Z.-H. (2008). Isolation Forest. *IEEE ICDM*, 413–422. | https://doi.org/10.1109/ICDM.2008.17 | Abstract |
| 2 | Hariri, S., Carrasco Kind, M., & Brunner, R. J. (2021). Extended Isolation Forest. *IEEE TKDE* (arXiv:1811.02141). | https://arxiv.org/abs/1811.02141 | Abstract |
| 3 | Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. *KDD '16*. | https://arxiv.org/abs/1603.02754 | Abstract |
| 4 | Lundberg, S. M., & Lee, S.-I. (2017). A unified approach to interpreting model predictions. *NeurIPS*. (SHAP; used only in the v1 prototype.) | https://arxiv.org/abs/1705.07874 | Abstract |
| 5 | Niculescu-Mizil, A., & Caruana, R. (2005). Predicting good probabilities with supervised learning. *ICML*, 625–632. | https://doi.org/10.1145/1102351.1102430 | Abstract |
| 6 | Roberts, D. R., et al. (2017). Cross-validation strategies for data with temporal, spatial, hierarchical, or phylogenetic structure. *Ecography*, 40(8), 913–929. | https://doi.org/10.1111/ecog.02881 | Abstract |
| 7 | Li, P., Abdel-Aty, M., & Yuan, J. (2021). Using bus critical driving events as surrogate safety measures for pedestrian and bicycle crashes based on GPS trajectory data. *Accident Analysis & Prevention*, 150, 105924. | https://doi.org/10.1016/j.aap.2020.105924 | Abstract |
| 8 | Longitudinal jerk and celeration as measures of safety in bus rapid transit drivers in Tehran (Cranfield repository). | https://dspace.lib.cranfield.ac.uk/handle/1826/15073 | Title / snippet (authors and year to be confirmed) |
| 9 | Edwards, K., & Hosein, P. (2025). A self-contained spatio-temporal anomaly detection application for travel safety. *2025 IEEE ICTMOD*, 1–6. | https://doi.org/10.1109/ICTMOD66732.2025.11371875 | **Full** |
| 10 | iBAT: Detecting anomalous taxi trajectories from GPS traces (UbiComp 2011); iBOAT: Isolation-based online anomalous trajectory detection (IEEE T-ITS, 2013). | https://hal.archives-ouvertes.fr/hal-01302049 | Abstract (authors to be confirmed) |
| 11 | Graph Enhanced Trajectory Anomaly Detection (arXiv:2509.18386). | https://arxiv.org/pdf/2509.18386 | Abstract |
| 12 | Towards physics-informed diffusion for anomaly detection in trajectories (arXiv:2506.06999). | https://arxiv.org/pdf/2506.06999 | Abstract |
| 13 | Unsupervised abnormal stop detection for long-distance coaches with low-frequency GPS (arXiv:2411.04422). | https://arxiv.org/html/2411.04422v1 | Abstract |
| 14 | Few-shot semi-supervised learning for abnormal stop detection from sparse GPS trajectories (arXiv:2510.12686). | https://ar5iv.labs.arxiv.org/html/2510.12686 | Abstract |
| 15 | Frank, E., & Hall, M. (2001). A simple approach to ordinal classification. *ECML 2001*, LNCS, 145–156. | https://link.springer.com/content/pdf/10.1007/3-540-44795-4_13.pdf | Abstract |
| 16 | Angelopoulos, A. N., & Bates, S. (2021). A gentle introduction to conformal prediction and distribution-free uncertainty quantification (arXiv:2107.07511). | https://arxiv.org/abs/2107.07511 | Abstract |
| 17 | Gibbs, I., & Candès, E. (2021). Adaptive conformal inference under distribution shift. *NeurIPS* (arXiv:2106.00170). | https://arxiv.org/abs/2106.00170 | Abstract |
| 18 | Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). On calibration of modern neural networks. *ICML* (arXiv:1706.04599). | https://arxiv.org/abs/1706.04599 | Abstract |
| 19 | Hybrid deep learning for anomaly detection in mental-healthcare provider billing (arXiv:2507.01924). | https://arxiv.org/pdf/2507.01924 | Abstract |
| 20 | AnomalyAID: Reliable interpretation for semi-supervised network anomaly detection (arXiv:2411.11293). | https://arxiv.org/pdf/2411.11293 | Abstract |
| 21 | Unsupervised learning for detection of rare driving scenarios (arXiv:2512.23585). | https://arxiv.org/html/2512.23585 | Abstract |
| 22 | Anomalous behavior detection in trajectory data of older drivers (arXiv:2311.17822). | https://arxiv.org/pdf/2311.17822 | Abstract |
| 23 | A comparative analysis of multiple methods for predicting a specific type of crime in the City of Chicago (arXiv:2304.13464). | https://arxiv.org/pdf/2304.13464 | Abstract |
| 24 | Data Science for Public Policy: felony prediction project (Georgetown course project; not peer reviewed). | https://troy-yu-cheng.github.io/final-project | Snippet |
| 25 | Individual, travel, and bus stop characteristics influencing travelers' safety perceptions (TU Delft; Stockholm). | https://research.tudelft.nl/en/publications/individual-travel-and-bus-stop-characteristics-influencing-travel | Abstract |
| 26 | Passenger Personal Security Report 2024 (National Transport Authority, Ireland). | https://www.nationaltransport.ie/wp-content/uploads/2025/04/Passenger-Personal-Security-Report-2024.pdf | Snippet |
| 27 | Modified feminist self-defense: violence against women in transit in Metro Manila (DLSU, PIDS webinar slides, 2023). | https://pidswebs.pids.gov.ph/CDN/document/March-2-2023-webinar-SERP-P-DLSU-Empowerment-Feminist-Self-Defense.pdf | Slides, method only |
| 28 | Hard-braking detection from GPS in cycling (*Sensors*; PMC8234598). Confirm this is the paper that reports the 1 Hz minimum before citing. | https://pmc.ncbi.nlm.nih.gov/articles/PMC8234598 | Snippet |
| 29 | Conformal prediction algorithms for time series forecasting: methods and benchmarking (arXiv:2601.18509). | https://www.alphaxiv.org/abs/2601.18509 | Abstract |
| 30 | Evaluating validity of synthetic data in perception tasks for autonomous vehicles (SJSU). Perception, not tabular models; context only. | https://scholarworks.sjsu.edu/faculty_rsca/26 | Snippet |
