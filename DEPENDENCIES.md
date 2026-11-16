# Dependencies and verified environments

The portable scripts derive all project paths from their own location. No author home directory, drive letter, network request or database account is required for the three supplied replay commands. Historical absolute paths remain as provenance strings in copied metadata and manifests; the portable archived-input reader locates cached graph bytes using the included relative structure.

## Result recomputation

Verified on Python 3.12.10, Windows 11, NumPy 2.2.6. Only NumPy and the Python standard library are required. `metrics.py` implements the original array metrics without a scikit-learn dependency. See `requirements-results.txt`.

## Tree checkpoint replay

Verified using the original Windows Python 3.12.10 tree-model environment:

- NumPy 2.2.6
- scikit-learn 1.7.1
- XGBoost 3.0.4
- joblib 1.5.2
- SciPy 1.16.1 and threadpoolctl, as dependencies of the original tree engine

The RF checkpoint files are serialized estimators; use the recorded matching scikit-learn version. The XGBoost checkpoints use the JSON model format. The runner imports the original array-only model engine from the bundled relative `research_plan/step09_2026-10-15/` path.

## Graph and request replay

Verified on Python 3.12.15, Linux/WSL CPU, PyTorch 2.8.0+cpu, NumPy 2.5.4. Graph checkpoints use the original `models.py` implementation; checkpoint metadata binds its exact source SHA256. PyTorch CPU inference uses four threads, one interop thread, deterministic algorithms and evaluation-mode graph embedding reuse. A GPU is not required. NumPy/PyTorch CPU builds and any platform-specific installation source should be selected for the target machine; no installation is performed by the replay scripts.

The actual tool versions for both backends are also embedded in the two replay validation JSON files. The different environment versions are recorded rather than represented as one previously tested shared environment.

## Boundaries

Full retraining is not part of the portable replay interface. Original training source is included for inspection, but its run orchestrator also checks historical selected-specification/original-checkpoint paths which are not all part of this closure. Do not invoke the original training orchestrator and interpret a missing historical path as an inference failure.

Raw source acquisition, data-standardization/identity construction, arbitrary new-structure feature extraction and fresh GPAW calculations require additional original research inputs/software and are not claimed to have been reproduced by these commands. The retained 78 graph caches and exact 770-column feature arrays are the explicit inputs to the verified checkpoint replay.

Original plotting scripts under `provenance_code/` reference local layout/QC helpers and some original analysis paths. Exact numeric sources and reference PDF/SVG figures are included. Portable regeneration of their original visual layout is not an acceptance claim here.
