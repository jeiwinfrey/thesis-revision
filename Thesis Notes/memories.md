# LIGTAS thesis memory

## Study goal

LIGTAS uses smartphone GPS and timestamps to identify unusual movement during a commute and forecast the highest concern level within the next 60 seconds. The output can indicate that a trip may need attention. It measures movement patterns, and GPS alone cannot establish whether a commuter is in danger.

The modeling pipeline is shared across commuters. For each test trip, the model is trained separately for that commuter using only their earlier completed trips on the monitored route. The current replay assumes one dataset folder per commuter and a known route and direction.

## Methodology outline

### Data gathering

- **Study population and sampling** — Recruit volunteers through criterion sampling, a form of purposive sampling. The study population is adults who regularly commute on selected public transit routes in Ilocos Norte. Commuters are the participants, and their recorded trips are the units used to train and evaluate the model.
- **Mobile data collection app** — Use this term in the thesis. The app records GPS data during trips and allows participants to enter trip notes. Model outputs are evaluated separately. App usability, battery consumption, alert delivery, and emergency response are outside the evaluation.
- **Recording trips** — Start recording at boarding and stop at alighting. Save the actual fix timestamp, latitude, and longitude. Horizontal accuracy is useful when available.
- **Organizing trip records** — Keep each commuter's trips separate. Save the expected route and travel direction with each trip.
- **Collecting trip notes** — Use a short post-trip check for whether anything unusual happened. Offer Yes, No, and Unsure. For Yes, ask what happened and when it began.

### Data preprocessing

- **Cleaning GPS records** — Remove invalid, duplicate, or implausible fixes. Flag gaps in coverage so missing data do not count as stopped time.
- **Measuring trip movement** — Calculate speed, stop duration, distance from the expected route, and progress along it using elapsed time.

### Model development

- **Scoring unusual movement** — Fit Isolation Forest on earlier trips from the same commuter and use it to score eligible GPS observations.
- **Assigning the current concern level** — Combine the unusualness score, distance from the expected route, and stop duration relative to typical stops along the route. The four levels are Normal, Minor, Major, and Critical.
- **Creating forecast labels** — For training observations with a full minute of future GPS coverage, use the highest concern level reached in that minute as the target.
- **Training the forecast model** — Fit XGBoost using information available up to the forecast time. Hold out recent trips from fitting for calibration.

### Model evaluation

- **Testing on later trips** — Replay completed trips in chronological order and evaluate how performance changes as each commuter's history grows.
- **Comparing forecasts** — Compare XGBoost with a persistence baseline, which assumes the current concern level stays the same. Logistic regression is another comparison. Use a binary confusion matrix for an alert threshold such as Major or Critical and, if needed, a four-class matrix.
- **Checking forecast probabilities** — Measure whether predicted probabilities agree with how often the corresponding model-defined concern levels occur.
- **Checking alerts against reviewed events** — If notes are collected, match alerts to independently documented unusual stops and route deviations. Count early warnings, late detections, missed events, and false alarms on confirmed ordinary trips.

## What trip notes change

Trip notes are for evaluation. The model makes its predictions from GPS and timestamps. A note that gives an event and approximate onset lets the evaluator check whether an alert preceded it. A documented event without an onset cannot establish warning lead time. Leave unreviewed or uncertain trips unknown.

There are two distinct confusion matrices. The forecast matrix compares predictions with the concern levels produced by the GPS rules. The field matrix compares alerts with reviewed event and ordinary trips. A false-positive trip counts once in the field matrix, even if it contains multiple false alarm episodes.

The field evaluation uses trip notes when available. Without reviewed events or another independent reference, keep the forecast matrix and report alert frequency for unreviewed data. Existing real traces can be reused when they meet the recording format. If the only available data are the generated trips, the field evaluation needs new commuter recordings.

## Implementation and evidence

- The audited model evaluation workflow is CSV replay. The mobile data collection app records GPS data and trip notes. It does not display model outputs.
- The revised replay handles timestamp units, gaps, complete forecast windows, return-route progress, local stop references, and separate calibration. The evaluator reports model-defined matrices and accepts reviewed labels for field evaluation.
- The supplied `01-test-route` dataset contains 60 generated trips. The revised no-expiry replay evaluated 2,342 full windows across 55 judged trips. At the 0.50 Major-or-higher cutoff, XGBoost caught 1 of 95 positive windows, logistic regression caught 25, and persistence caught 39. These are development results. Compare early-warning lead and false-alert rates on reviewed field trips before selecting an alert model.
- See [the algorithm and methodology notes](../LIGTAS-Model/notes/ALGORITHM.md), [the replay report](../LIGTAS-Model/results/evaluation_v4.html), and [the trip-notes template](../LIGTAS-Model/results/labels_template.csv).

## Thesis writing preferences

Keep the Background of the Study close to the original wording and leave technical details for the methodology. Explain personalization as training separately for each commuter on their earlier completed trips. In the specific objective, use one sentence, keep the word “trip” where it adds clarity, and avoid repeating it several times. State that XGBoost forecasts the highest concern level in the next minute.

Use “GPS-derived movement features” as the umbrella term for speed, stop duration, and distance from the usual route. Keep the 10-second recording interval and cleaning procedures in the methods chapters, while retaining data cleaning in Objective 1. Keep each research question focused and use “How” once per question. Introduce the specific objectives with “Specifically, this study seeks to:” and start each item with a lowercase verb. In the Significance of the Study, list researchers first, then software developers and engineers, future commuters, and families and guardians. Write Scope and Delimitations as one section. Limit the findings to the participants, routes, and travel conditions studied.

Working evaluation objective, if the study collects reviewed notes:

> Evaluate model performance as each commuter’s travel history grows by training only on their earlier trips and testing on later ones, assessing forecast accuracy against a simple baseline, probability calibration, detection of documented unusual movement, and the false alarm rate on confirmed ordinary trips.

## Chapter 2 revision decisions

The working Word copy is [Thesis Revised Chapter 2.docx](../Thesis%20Revised%20Chapter%202.docx). Keep the prose straightforward, with short sentences and clear explanations. The opening paragraph goes directly under the chapter title without a heading or section number. Use this approved opening:

> This chapter reviews studies on commuter safety and the use of GPS data to detect unusual trip movement. It explains the research gap that motivates the study and reviews how earlier studies collected and cleaned GPS data, measured stops and route deviations, and used those measures to build and evaluate models. It also explains why a reading can be unusual for a trip without showing that a commuter is in danger.

- **2.1 Study Context and Research Gap** groups **2.1.1 Commuter Safety in the Philippine Context** and **2.1.2 Related Studies and Research Gap**. Connect the gap to prior work on route deviations, unusual stops, and forecasting.
- **2.2 Data Gathering** covers **2.2.1 Recording Trips**, **2.2.2 Organizing Trip Records**, **2.2.3 Collecting Optional Trip Notes**, **2.2.4 Sampling Method**, **2.2.5 Participant Count**, **2.2.6 Trip Sample Size**, and **2.2.7 Recording Period**. The separate **2.2.8 Location Quality and Privacy** subsection was removed.
- **2.3 Data Preprocessing** covers GPS cleaning, recording gaps, and movement measures. **2.4 Model Development** covers Isolation Forest scoring, concern levels and thresholds, next-minute labels, and XGBoost forecasting. **2.5 Model Evaluation** covers later-trip testing, the persistence baseline and confusion matrix, probability calibration, and alert checks against optional trip notes. **2.6 Conceptual Framework** shows the overall workflow. Omit standalone tools sections.

### Field collection targets and sources

- Use criterion sampling. Eligible participants are consenting adults who regularly ride a selected route and have a compatible smartphone. Etikan et al. (2016) and [Palinkas et al. (2015)](https://doi.org/10.1007/s10488-013-0528-y) support the sampling approach. [Ankunda and Venter (2025)](https://doi.org/10.1016/j.tbs.2024.100936) provide a public transport GPS example.
- The proposed field target is **five commuters**, **60 trips per commuter**, **30 trips per direction**, and **300 planned trips total**. [Liu et al. (2013)](https://doi.org/10.1155/2013/615282) used five volunteers with GPS-equipped cars over nine weeks to evaluate a model of individual travel patterns. This supports the feasibility of a small group with repeated observations. Five is a study design choice for an initial field evaluation. Each commuter's history is analyzed separately.
- Keep the trip-count calculation in Chapter 2. Using [Viechtbauer et al. (2015)](https://doi.org/10.1016/j.jclinepi.2015.04.014), an assumed event probability of 5% per independent trip gives a 95% chance of seeing at least one event when n ≥ ln(0.05) / ln(0.95) ≈ 58.40. Round up to 59, then set the target to 60 for equal coverage in both directions. The 5% rate is a planning assumption. Repeated trips may be dependent, and the calculation concerns observing at least one event.
- Plan about **30 recording days** at one trip per direction per day. This also allows comparison of 7-day, 14-day, and full-history settings. [Winkler et al. (2026)](https://doi.org/10.1007/s11116-024-10517-1) provide a four-week smartphone GPS diary precedent.
- Chapter 3 states the field targets and recording setup and points back to Chapter 2 for the rationale. The completed initial replay used **60 synthetic trips**, **2,936 GPS readings at 10-second intervals**, one simulated commuter profile, and **zero field participants**.

### Route history and interpretation

Figure 1 treats GPS trip records as the input. Repeated GPS trips teach the model what movement is usual for each commuter. Isolation Forest scores new readings against the learned pattern. There is no separately supplied route reference in the conceptual framework. Distinguish the mapped line used in the initial replay and fixed-distance comparison from the learned movement pattern when reading earlier route-reference language in these notes.

Isolation Forest scores current unusualness. XGBoost forecasts the highest model-defined concern level in the next minute. Normal through Critical describe movement under the model's rules. The reviewed literature supplies no validated personal-danger threshold for these routes. Use independently reviewed trip notes for field event checks when available.
