# LIGTAS GPS movement model and evaluation

LIGTAS scores how unusual the movement of a trip looks for a commuter on a known route. It forecasts the maximum model-defined concern level over the next minute. The supplied dataset has 60 generated trips. The replay demonstrates the pipeline and provides a development benchmark.

## Collecting field data

- Use one dataset folder per commuter and one saved route per folder. Put the direction-specific polylines in `project.json`, ordered in the direction of travel. Place one CSV per in-vehicle trip under `a-to-b/` or `b-to-a/`.
- Start recording at boarding and stop at alighting. Keep the actual location-fix timestamp in ISO 8601 with a time-zone offset. Each CSV needs `timestamp,latitude,longitude`. An optional `accuracy` column holds the phone's horizontal accuracy in meters.
- Existing real GPS traces can be replayed if they meet this format. New GPS collection is needed if only generated data exist or the existing real traces omit required trips or routes. A trip-notes review can be added to existing real traces when independent evidence is available.
- Copy [the notes template](../results/labels_template.csv) and replace its file names with the relative CSV paths in the field dataset. Use `incident=0` for a reviewed trip with no qualifying event and leave it blank when unknown. For an independently observed event, use `incident=1`, `event_type=unexpected_stop` or `route_deviation`, and the ISO 8601 onset in `incident_time`. Add `end_time` when known. Use one row per event. Keep notes of other concerns distinct from these movement labels. Reviewers should apply a written rubric without seeing alerts.

## Model

1. Clean invalid coordinates and timestamps, duplicate fixes, implausible speed or acceleration, and optional poor accuracy. Reject trips with fewer than 20 accepted fixes, more than 20% rejected fixes, or no continuous pair of fixes. A gap longer than 25 seconds breaks continuity by default. This is a design choice for the 10-second study cadence and can be set with `--max-gap`.
2. Compute movement features over real elapsed time. Stop duration starts at zero, resets across missing coverage, and uses observed low-motion intervals. Progress increases in the direction of travel. The first fix and first fix after a gap lack movement history and receive no forecast.
3. Fit the context on earlier completed trips from the same commuter. Compare dwell to completed stops by road segment and direction when at least three trips support the reference, with direction and pooled fallbacks. The planned route remains valid. After five fitting trips in the direction, a route variant can also be accepted when supported by at least two distinct trips. These support cutoffs are design choices.
4. Combine Isolation Forest, route distance, and local dwell unusualness into four ordered concern levels. The robust z-score cutoffs and distance bands are model design parameters. Their severity labels describe movement, with no verified mapping to personal danger.
5. Fit cumulative XGBoost classifiers for Minor+, Major+, and Critical+. Use only full, observed future windows for training targets. Hold recent trips out of upstream fitting. Fit Platt scaling when the earlier calibration subset supports it and estimate empirical prediction sets on the last calibration trip. Overlapping windows from one trip are dependent, so the nominal 90% target is not a guarantee. The default XGBoost fit uses 120 trees without early stopping.
6. Replay each trip using only previously completed trips. Keep no-expiry pseudo-labels fixed across expiry settings. The forecast outputs and current scores may still change with the history setting.

## Confusion matrices

The report includes binary matrices for Minor+, Major+, and Critical+, an ordinal four-class matrix, and a separate table for forecasts made while the current concern is below the chosen alert level. Rows are reference labels and columns are predictions. The per-window reference is the future model-defined concern level, and the reported sample excludes partial horizons and GPS gaps.

A field matrix is available when reviewed notes are supplied. It counts a verified event trip as warned when an alert episode starts before an observed event within the forecast horizon. Verified ordinary trips with any alert count as false positives. Match each alert to at most one event. Report late detections, missed and unscored events, unmatched alerts, and lead time separately. An event with no observed pre-onset window is unscored. The evaluator records false alarms only on explicitly reviewed ordinary trips and compares the forecast with current-level alerts and a fixed route-distance rule. Use a later field split for reporting after choosing thresholds on development trips.

## Run the replay

From the workspace root, with the Python environment installed from [the dependency pins](../code/requirements.txt):

```sh
python LIGTAS-Model/code/evaluate_v4.py 01-test-route --out-dir LIGTAS-Model/results
```

Use the field dataset folder instead of `01-test-route` and add `--labels /path/to/reviewed_notes.csv --out-dir /path/to/field_results` for the field evaluation. To compare logistic regression, run the same evaluation in a separate output folder with `--model-opts '{"kind":"logistic"}'`. The JSON records parameters and package versions. The HTML shows the matrices. The CSV retains excluded fixes with their observation flag and empty target where the horizon cannot be scored.

## Aligning the thesis methodology

In Chapter 3, update the data collection and cleaning sections to describe optional accuracy and gap handling. Update Sections 3.5.1–3.5.6 to describe the local stop reference, supported route variants, complete forecast windows, and separate calibration. Update Sections 3.6.2 and 3.6.6–3.6.9 to separate pseudo-label agreement from independently reviewed field events. Describe the fixed-distance baseline as implemented and keep controlled route familiarization tests separate from field results. The current interface is a CSV replay.

## 9. References (as in Chapter 2)

- Deng, J., Pang, J., Xu, J., & Yu, H. (2024). *Unsupervised abnormal stop detection for long distance coaches with low-frequency GPS*. arXiv. https://arxiv.org/abs/2411.04422
- Edwards, K., & Hosein, P. (2025). A self-contained spatio-temporal anomaly detection application for travel safety. *2025 IEEE ICTMOD*. https://doi.org/10.1109/ICTMOD66732.2025.11371875
- Frank, E., & Hall, M. (2001). A simple approach to ordinal classification. *ECML 2001*. https://doi.org/10.1007/3-540-44795-4_13
- Gama, J., Sebastião, R., & Rodrigues, P. P. (2013). On evaluating stream learning algorithms. *Machine Learning, 90*(3), 317–346. https://doi.org/10.1007/s10994-012-5320-9
- Gama, J., Žliobaitė, I., Bifet, A., Pechenizkiy, M., & Bouchachia, A. (2014). A survey on concept drift adaptation. *ACM Computing Surveys, 46*(4), Article 44. https://doi.org/10.1145/2523813
- Liu, F. T., Ting, K. M., & Zhou, Z.-H. (2008). Isolation forest. *ICDM 2008*. https://doi.org/10.1109/ICDM.2008.17
- Chen, T., & Guestrin, C. (2016). XGBoost. *KDD '16*. https://doi.org/10.1145/2939672.2939785
- Leys, C., Ley, C., Klein, O., Bernard, P., & Licata, L. (2013). Detecting outliers: Do not use standard deviation around the mean, use absolute deviation around the median. *Journal of Experimental Social Psychology, 49*(4), 764–766.
- NIST/SEMATECH. (n.d.). *Detection of outliers*. https://itl.nist.gov/div898/handbook/eda/section3/eda35h.htm
- Bagheri, M., Mladenović, M. N., Kosonen, I., Nurminen, J. K., Roncoli, C., & Ylä-Jääski, A. (2020). A computational framework for revealing competitive travel times with low-carbon modes based on smartphone data collection. *Journal of Advanced Transportation, 2020*, Article 4693750.
- Mears, M., Brindley, P., Barrows, P., Richardson, M., & Maheswaran, R. (2021). Mapping urban greenspace use from mobile phone GPS data. *PLOS ONE, 16*(7), e0248622.
- Hyndman, R. J., & Athanasopoulos, G. (2021). *Forecasting: Principles and practice* (3rd ed.). OTexts.
- Niculescu-Mizil, A., & Caruana, R. (2005). Predicting good probabilities with supervised learning. *ICML 2005*.
- Sabir, M. A., Pang, J., Wu, J., & Ashraf, F. (2025). *Few shot semi-supervised learning for abnormal stop detection from sparse GPS trajectories*. arXiv. https://arxiv.org/abs/2510.12686
- Verma, R., Mittal, S., Lei, Z., Chen, X., & Ukkusuri, S. V. (2024). Comparison of home detection algorithms using smartphone GPS data. *EPJ Data Science, 13*, Article 6. https://doi.org/10.1140/epjds/s13688-023-00447-w

The full reference list is in Chapter 2.
