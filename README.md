# Janus/HetDB bilayer band-gap code, data and trained models

This repository supports **Probabilistic Applicability Screening and Residual Band-Gap Prediction for 2D Heterostructures from Monolayer Inputs**, the current fixed-configuration Janus/HetDB manuscript. It is publicly hosted at [whongmin51-crypto/kaeriMiChi](https://github.com/whongmin51-crypto/kaeriMiChi) (no separate DOI is assigned). Original code and derived documentation in this repository are covered by the root MIT `LICENSE`; third-party materials retain their original terms and provenance. Paper LaTeX source and manuscript compilation files are excluded.

The current analysis uses 11,561 fit records, disjoint calibration pools of 477/437/341 records, 742 final holdouts and 60 separately reserved source-diagnostic records. Related configurations remain grouped. The fixed configurations and their development provenance are retained in `data/provenance/protocol.json`; this package does not redefine the historical model-selection experiment.

## What can be run

Run commands from this directory, or pass the full path to the scripts. Outputs are written to `recomputed/` or `replayed/`; the retained evidence is read-only input.

### 1. Complete holdout-result recomputation

```text
python -m pip install -r requirements-results.txt
python recompute_results.py
```

This recomputes the published service's regression, probability classification, acceptance risk, processing/acceptance coverage, prediction-interval coverage, 1,000 composition-group bootstrap intervals and paired differences for **all 742 final holdouts and all 60 source-diagnostic records**. It also checks all 14 predefined candidate ensembles against their recorded raw-head comparisons. It verifies the input-copy hashes, then compares 6,887 and 2,042 scalar values, respectively, to the retained evaluation files. The supported holdout denominator is 721; the 21 abstentions remain in the original 742-record coverage denominator.

This command uses retained fixed predictions and evaluated targets. It does not train, select, recalibrate or run a neural-network checkpoint. A portable copy of the original pure-array metrics and bootstrap functions is included, with only evaluation I/O redirected into this directory. `recomputed/validation.json` records the actual verification.

### 2. All 70 checkpoint models on their archived inputs

This repository includes `model_bundle/` (about 821 MiB uncompressed). The small result-only ZIP excludes that directory. Preserve its relative `research_plan/` structure when obtaining or depositing it.

```text
python replay_checkpoints.py --backend trees
python replay_checkpoints.py --backend graph
```

The commands independently replay 30 RF/XGBoost models and 40 CGCNN/GANN models on every calibration and holdout role: **2,057 input rows per model**. The five-seed means are compared with all archived ensemble heads using absolute and relative tolerances of 1e-7. Each checkpoint and input is bound to its retained SHA256 record. Model architectures, graph math and preprocessing source files are copied without changes.

Actual verification used the original Windows tree-model environment and the original Linux/WSL CPU PyTorch environment, respectively. Both passed. A single unified environment and a different machine have not been verified. Exact package versions and per-ensemble maximum differences are recorded in `replayed/trees_validation.json` and `replayed/graph_validation.json`; install guides are in `DEPENDENCIES.md`.

### 3. Six native-reference requests through the calibrated service

```text
python replay_requests.py
```

This uses the selected 15 graph checkpoints, the fitted probability map, decision threshold, residual interval and the original frozen input checks. Six archived retrospective application requests and their layer-graph NPZ files are included. Only graph path spellings are relocated; the 770 numerical input features and graph bytes are unchanged. Predicted gaps, probabilities, raw heads and interval endpoints agree with the retained application records within 1e-7. Missing contact edges, SOC mismatch, true twist and matched-strain input are rejected or withheld as specified. The contract output is redirected to `replayed/` to avoid modifying packaged source files.

The supplied requests begin with derived raw-feature values and graph NPZ arrays. These tests do not establish arbitrary CIF-to-feature conversion, database re-download/reconstruction, full retraining, fresh DFT execution or cross-machine reproduction. The six applications are retrospective evaluations of existing DFT targets.

## Files

- `data/`: immutable copies of current ensemble predictions, evaluation targets/metadata, calibration records, primary predictions, application cases and provenance.
- `code/`: pure-array metrics and bootstrap/evaluation functions used by the portable recomputation.
- `source_data/`: exact figure data and the conditional-performance analysis data used in the current manuscript.
- `reference_figures/`: the retained PDF/SVG figures for numerical/visual comparison.
- `provenance_code/`: original plotting scripts, retained as provenance. These original scripts may reference the original project layout and local figure-QA helpers; they are not advertised as portable executables.
- `model_bundle/`: 70 current fitted checkpoints, complete fitted input arrays and state, 78 monolayer graph caches, the necessary original inference source modules, and six portable service requests.
- `copy_manifest.json`: origin/size/hash of each copied result-evidence file; `model_bundle/copy_manifest.json` records the model bundle.
- `optional_model_manifest.json`: checkpoint inventory with original source paths, sizes and hashes.
- `recomputed/` and `replayed/`: actual local verification outputs, with no new fitting or scientific decisions.

See `DATA_DICTIONARY.md`, `DEPENDENCIES.md`, `RIGHTS_AND_RELEASE_PENDING.md` and `repository_validation.json` for the data semantics, executed scope and licensing/provenance notes. Original scientific records and completion gates were not edited. This repository provides reproducibility code, data and model artifacts, not a replacement scientific completion gate.

## Repository deposit and provenance

The original staging directory and scientific records remain unchanged. `github_deposit_manifest.json` records the initial repository deposit, its source hashes and intentional exclusions. `repository_validation.json` and the original manifests are retained evidence of the earlier local verification; their original hosting fields and local cache entries describe that historical snapshot. Python caches are excluded from Git.

Six additional original research source files are retained under `model_bundle/research_plan/`: current input preparation and calibration/evaluation, the original calibration utilities, and feature construction. These files preserve the research implementation and may depend on the original project layout and historical parameter specifications. They do not establish portable full retraining or arbitrary raw-structure processing. The supported executable reproduction commands and their limits are listed above.
