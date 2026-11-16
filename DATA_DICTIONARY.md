# Data dictionary

## Evaluation populations

`data/evaluation/{sealed_final_test,external_validation_diagnostic}/evaluated_predictions.jsonl` has one JSON object per evaluated input. Each object contains:

- `metadata`: source (`Janus` or `HetDB`), record ID, physical-material-pair ID, composition group, reference track, operational PBE target `g_PBE_eV`, contact-side Anderson value `g_A_eV`, signed residual `r_eV` and binary baseline-failure label `y`.
- `frozen_prediction`: support status, calibrated and raw failure probability, corrected gap, predefined direct-gap and routed comparator outputs, and acceptance decision when supported.

The failure label is 1 exactly when the original Anderson absolute error is strictly greater than 0.100 eV. Equality belongs to the applicable class. All gaps/residuals are in eV. Targets are the nonnegative operational source gap. The metadata preserves original source-evidence qualifications and hashes.

An `out_of_domain` record has no service gap/probability. It still counts in processing and unconditional coverage denominators. Raw candidate benchmark files may contain numerical outputs for such a record; they do not override the service abstention.

## Weighting and intervals

Physical pairs receive equal weight within each source; configurations share their pair's weight; source-macro statistics average the source-specific statistics. Source-macro RMSE is the mean of the source-specific RMSE values, not the square root of the pooled mean square. Both quantities are separately retained.

Composition groups are the synchronized resampling unit for 1,000 fixed-model bootstrap replicates with seed 20261025. Interval calibration uses the maximum absolute error within each calibration composition group, nominal 90% coverage and nonnegative lower-endpoint projection. Nominal coverage is distinct from observed group/row coverage; all are retained in the result files.

## Ensembles and calibration

`data/inference/<role>/<candidate>_ensemble.jsonl` contains five-seed means. Independent heads use `raw_head`; joint models retain `p_fail`, `predicted_residual_eV` and `predicted_direct_raw_eV`. Residual gap prediction is `max(0, g_A + raw residual)`; direct gap prediction is `max(0, raw direct gap)`.

`data/calibration/` retains the fitted monotone probability map, frozen acceptance cutoff, interval radii and calibration predictions. These objects are copied and applied; they are not re-estimated by any provided replay command.

## Figure source data

- Figure 1: explanatory workflow diagram, retained in `reference_figures/our_task.{pdf,svg}`. It is not measured data.
- Figure 2: `source_data/figures/model_comparison.samples.csv`, `model_comparison.cohort.csv`, numerical summaries, statistics and paired differences. The common supported cohort is 721 records.
- Figure 3: `source_data/figures/source_errors_source.csv` contains the 12 source/pipeline MAEs and exact 95% interval limits; `source_errors.jsonl` mirrors the plotted values.
- Figure 4: `source_data/figures/pair_response.jsonl` contains 40 physical pairs, true/predicted residual ranges and whether the original applicability class varies within the pair.
- Figure 5: `source_data/figures/controlled_response_source.csv` contains 20 intervention contrasts in four independent confirmation pairs. DFT gap response equals isolated-layer-reference response plus residual response. Original numerical envelopes and their measured/unmeasured status remain distinct.
- Supplementary training trajectories: `source_data/training_curves/` contains one retained training-curve JSONL file per residual-model seed. No new fits were run.
- Risk/coverage source: the retained threshold-by-threshold curves are in `data/evaluation/`.
- Section 3.6 conditional-performance data: exact samples/assignments and stratified result tables appear in `source_data/conditional_performance/`; this post-freeze descriptive analysis is not a new model-selection exercise.

## Model input arrays

`model_bundle/research_plan/retrain_no_tony_2026-11-02/inputs/raw_features.npz` stores ordered record IDs, 770 ordered feature names and raw feature values. Null/missing values retain NumPy NaN semantics. The fit-only state supplies medians, population scales and fixed missing/constant-channel rules, yielding 1,540 model channels. `graph_pairs.json` maps each input ID to its lower/upper monolayer graph IDs. The 78 NPZ caches contain atomic numbers, edge indices, 32-RBF edge values and periodic edge vectors. `graph_manifest.json` binds their bytes by SHA256.

Record roles are immutable and are listed in `role_ids.json`. No script transfers evaluation records into fitting or recalibration.
