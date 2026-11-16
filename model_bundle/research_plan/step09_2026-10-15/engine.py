"""Row-equal RF/XGBoost fits for the frozen Step07/08 experiment.

The engine receives arrays only. It has no dataset/label loader and cannot open
calibration, external-validation or test labels. Iteration selection belongs to
the explicit inner-stop arguments; an outer refit is a new call and estimator.
"""
from __future__ import annotations

import os

# Set these before importing NumPy and retain explicit threadpool limits below.
for _key in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "BLIS_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "1"
os.environ["OMP_NUM_THREADS"] = "4"

from collections import Counter
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import platform
import time
from typing import Any

import joblib
import numpy as np
import sklearn
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (average_precision_score, balanced_accuracy_score,
                             confusion_matrix, roc_auc_score)
from threadpoolctl import threadpool_limits
import xgboost as xgb

TASKS = ("classifier", "residual_regression", "direct_gap_regression")
FAMILIES = ("RF", "XGBoost")
N_THREADS = 4


@dataclass
class FitResult:
    estimator: Any
    metadata: dict
    curve: dict


def _vector(x, n=None, name="vector", dtype=np.float64):
    a = np.asarray(x, dtype=dtype)
    if a.ndim != 1 or (n is not None and len(a) != n):
        raise ValueError(f"Invalid {name} shape: {a.shape}")
    if dtype is not str and not np.isfinite(a).all():
        raise ValueError(f"Non-finite {name}")
    return a


def source_pair_weights(source, pair):
    """Weights sum to one: source macro -> pair macro -> row mean."""
    source = _vector(source, name="source", dtype=str)
    pair = _vector(pair, len(source), "pair", str)
    if not len(source):
        raise ValueError("No rows for evaluation")
    n_sources = len(set(source))
    pair_counts = Counter(zip(source, pair))
    pairs_per_source = Counter(s for s, _ in pair_counts)
    return np.asarray([1.0 / (n_sources * pairs_per_source[s] * pair_counts[(s, p)])
                       for s, p in zip(source, pair)], dtype=np.float64)


def project_gap(task, head, g_A=None):
    """Apply the same physical nonnegative projection at every scored stage."""
    pred = _vector(head, name="head")
    if task == "residual_regression":
        pred = pred + _vector(g_A, len(pred), "g_A")
    elif task != "direct_gap_regression":
        raise ValueError("Gap projection requires a regression task")
    return np.maximum(0.0, pred)


def _metric_context(task, y, g_A, source, pair):
    y = _vector(y, name="target")
    w = source_pair_weights(source, pair)
    if len(w) != len(y):
        raise ValueError("Target/evaluation identity length mismatch")
    if task == "classifier":
        if not np.isin(y, [0, 1]).all():
            raise ValueError("Classifier targets must be 0/1 failure labels")
        return y, None, w
    a = _vector(g_A, len(y), "g_A") if task == "residual_regression" else None
    return y, a, w


def _protocol_metric(task, pred, context):
    y, a, w = context
    pred = _vector(pred, len(y), "prediction")
    if task == "classifier":
        p = np.clip(pred, 1e-15, 1 - 1e-15)
        return float(np.dot(w, -(y * np.log(p) + (1-y) * np.log1p(-p))))
    truth_gap = y + a if task == "residual_regression" else y
    return float(np.dot(w, np.abs(project_gap(task, pred, a) - truth_gap)))


def weighted_metrics(task, y, head, g_A, source, pair):
    """All-source macro metrics and individual-source pair-weighted metrics.

    Classification AP/AUROC are computed inside each source and then averaged;
    pooling rows with macro weights is not equivalent for those rank metrics.
    Single-class rank scores are null and AP follows sklearn's definition when
    positives exist. Undefined scores are explicit instead of made-up zeros.
    """
    if task not in TASKS:
        raise ValueError(task)
    y = _vector(y, name="target")
    p = _vector(head, len(y), "prediction")
    sources = _vector(source, len(y), "source", str)
    pairs = _vector(pair, len(y), "pair", str)
    a = _vector(g_A, len(y), "g_A") if task == "residual_regression" else None
    by_source = {}
    for s in sorted(set(sources)):
        k = sources == s
        yt, pt = y[k], p[k]
        wt = source_pair_weights(sources[k], pairs[k])
        d = {"rows": int(k.sum()), "physical_pairs": int(len(set(pairs[k])))}
        if task == "classifier":
            if not np.isin(yt, [0, 1]).all() or np.any((pt < 0) | (pt > 1)):
                raise ValueError("Invalid classification labels/probabilities")
            pc = np.clip(pt, 1e-15, 1-1e-15)
            hard = pt >= 0.5  # diagnostic threshold only; no calibrated route
            cm = confusion_matrix(yt, hard, labels=[0, 1], sample_weight=wt)
            tn, fp, fn, tp = cm.ravel()
            recall = float(tp/(tp+fn)) if tp+fn else None
            precision = float(tp/(tp+fp)) if tp+fp else None
            f1 = float(2*tp/(2*tp+fp+fn)) if 2*tp+fp+fn else None
            d.update({
                "AP": float(average_precision_score(yt, pt, sample_weight=wt)) if np.any(yt == 1) else None,
                "AUROC": float(roc_auc_score(yt, pt, sample_weight=wt)) if len(set(yt)) == 2 else None,
                "log_loss": float(np.dot(wt, -(yt*np.log(pc)+(1-yt)*np.log1p(-pc)))),
                "Brier": float(np.dot(wt, (pt-yt)**2)),
                "accuracy_at_0p5": float(np.dot(wt, hard == yt)),
                "balanced_accuracy_at_0p5": float(balanced_accuracy_score(yt, hard, sample_weight=wt)) if len(set(yt)) == 2 else None,
                "failure_recall_at_0p5": recall,
                "failure_precision_at_0p5": precision,
                "failure_F1_at_0p5": f1,
                "confusion_at_0p5_pair_weighted": cm.tolist(),
            })
        else:
            ag = a[k] if a is not None else None
            truth = yt + ag if a is not None else yt
            pred = project_gap(task, pt, ag)
            error = pred-truth
            denom = float(np.dot(wt, (truth-np.dot(wt, truth))**2))
            d.update({"MAE_eV": float(np.dot(wt, abs(error))),
                      "RMSE_eV": float(np.sqrt(np.dot(wt, error**2))),
                      "bias_eV": float(np.dot(wt, error)),
                      "R2": 1-float(np.dot(wt, error**2))/denom if denom > 0 else None})
        by_source[str(s)] = d
    keys = ["AP", "AUROC", "log_loss", "Brier", "accuracy_at_0p5", "balanced_accuracy_at_0p5",
            "failure_recall_at_0p5", "failure_precision_at_0p5", "failure_F1_at_0p5"] if task == "classifier" else ["MAE_eV", "RMSE_eV", "bias_eV", "R2"]
    macro = {key: float(np.mean([d[key] for d in by_source.values()]))
             if all(d[key] is not None for d in by_source.values()) else None for key in keys}
    return {"task": task, "by_source": by_source, "macro": macro,
            "row_weighting": "equal physical pair within source; equal source macro",
            "classification_threshold": 0.5 if task == "classifier" else None,
            "classification_threshold_usage": "uncalibrated diagnostic only" if task == "classifier" else None}


def selection_score(task, y, head, g_A, source, pair):
    """Higher AP is better for classification; lower clipped-gap MAE for regressions."""
    metrics = weighted_metrics(task, y, head, g_A, source, pair)
    score = metrics["macro"]["AP" if task == "classifier" else "MAE_eV"]
    if score is None:
        raise ValueError("Primary source-macro selection score is undefined")
    return float(score)


def dependency_versions():
    return {"python": platform.python_version(), "numpy": np.__version__,
            "sklearn": sklearn.__version__, "xgboost": xgb.__version__,
            "joblib": joblib.__version__}


def fit_model(family, task, params, seed, X, y, g_A=None, source=None, pair=None,
              early_stop=None, iterations=None, output_path=None):
    """Fit a fresh estimator; ``y`` is y_fail, residual r, or PBE gap by task.

    early_stop = {X, y, g_A, source, pair} is accepted only for XGBoost.
    Omit it for the full outer/final refit, supplying inner-selected iterations.
    The caller is responsible for the frozen identities and preprocessing scope.
    No sample weights are passed to either fitting algorithm.
    """
    if family not in FAMILIES or task not in TASKS:
        raise ValueError("Unsupported model family/task")
    X = np.asarray(X, dtype=np.float32, order="C")
    if X.ndim != 2 or not np.isfinite(X).all():
        raise ValueError("Features must be a finite two-dimensional matrix")
    y = _vector(y, len(X), "target")
    if not len(X):
        raise ValueError("Cannot fit an empty cohort")
    context = _metric_context(task, y, g_A, source, pair)
    p = dict(params)
    timer = time.perf_counter()
    curve = {}
    meta = {"family": family, "task": task, "seed": int(seed),
            "requested_params": dict(params), "sample_weight_mode": "row_equal",
            "device": "CPU", "model_threads": N_THREADS, "blas_threads": 1,
            "n_fit_rows": len(X), "n_features": X.shape[1],
            "dependency_versions": dependency_versions(),
            "early_stop_used": early_stop is not None,
            "outer_validation_used_for_iteration_selection": False}
    with threadpool_limits(limits=1, user_api="blas"):
        if family == "RF":
            if early_stop is not None:
                raise ValueError("RF has no early-stopping pilot in the protocol")
            if iterations is not None and int(iterations) != int(p["n_estimators"]):
                raise ValueError("RF tree count is its frozen candidate parameter")
            if "class_weight" in p and p["class_weight"] is not None:
                raise ValueError("Main RF fit must remain row-equal")
            model_class = RandomForestClassifier if task == "classifier" else RandomForestRegressor
            estimator = model_class(**p, random_state=int(seed), n_jobs=N_THREADS)
            estimator.fit(X, y)
            head = predict_head(estimator, family, task, X)
            curve = {"fit": {"tree_count": [int(p["n_estimators"])],
                             "protocol_metric": [_protocol_metric(task, head, context)]},
                     "curve_semantics": "one completed forest; RF has no iterative early-stop trajectory"}
            meta.update({"selected_iterations": int(p["n_estimators"]),
                         "trained_iterations": int(p["n_estimators"]),
                         "saved_iterations": len(estimator.estimators_),
                         "native_params": estimator.get_params()})
        else:
            maximum = int(p.pop("n_estimators_max"))
            patience = int(p.pop("early_stopping_rounds"))
            objective = "binary:logistic" if task == "classifier" else "reg:squarederror"
            if p.get("objective", objective) != objective:
                raise ValueError("Frozen objective does not match task")
            p.update({"objective": objective, "tree_method": "hist", "device": "cpu",
                      "nthread": N_THREADS, "seed": int(seed), "verbosity": 0,
                      "eval_metric": "logloss" if task == "classifier" else "rmse"})
            n_rounds = maximum if early_stop is not None else int(iterations if iterations is not None else maximum)
            if not 1 <= n_rounds <= maximum:
                raise ValueError("Refit rounds exceed the frozen candidate maximum")
            dtrain = xgb.DMatrix(X, label=y, nthread=N_THREADS)
            matrices = [(dtrain, "inner_fit" if early_stop is not None else "fit")]
            contexts = {dtrain.handle.value: context}
            callbacks = []
            if early_stop is not None:
                if iterations is not None:
                    raise ValueError("A pilot chooses iterations; do not preset them")
                stop_x = np.asarray(early_stop["X"], dtype=np.float32, order="C")
                if stop_x.ndim != 2 or stop_x.shape[1] != X.shape[1] or not np.isfinite(stop_x).all():
                    raise ValueError("Invalid inner-stop feature matrix")
                stop_y = _vector(early_stop["y"], len(stop_x), "inner_stop target")
                ds = xgb.DMatrix(stop_x, label=stop_y, nthread=N_THREADS)
                matrices.append((ds, "inner_stop"))
                contexts[ds.handle.value] = _metric_context(task, stop_y, early_stop.get("g_A"),
                                                           early_stop["source"], early_stop["pair"])
                callbacks = [xgb.callback.EarlyStopping(rounds=patience, metric_name="protocol_metric",
                                                        data_name="inner_stop", maximize=False,
                                                        save_best=True, min_delta=0.0)]

            def custom_metric(pred, dm):
                return "protocol_metric", _protocol_metric(task, pred, contexts[dm.handle.value])

            estimator = xgb.train(p, dtrain, num_boost_round=n_rounds, evals=matrices,
                                  evals_result=curve, custom_metric=custom_metric,
                                  callbacks=callbacks, verbose_eval=False)
            rounds_run = len(curve[matrices[0][1]]["protocol_metric"])
            chosen = int(estimator.best_iteration)+1 if early_stop is not None else n_rounds
            meta.update({"selected_iterations": chosen, "trained_iterations": rounds_run,
                         "saved_iterations": int(estimator.num_boosted_rounds()),
                         "n_inner_stop_rows": len(stop_x) if early_stop is not None else 0,
                         "native_params": p, "n_estimators_max": maximum,
                         "early_stopping_rounds": patience if early_stop is not None else None,
                         "best_inner_score": float(estimator.best_score) if early_stop is not None else None,
                         "protocol_metric": "source/pair weighted binary logloss" if task == "classifier" else "source/pair weighted reconstructed nonnegative gap MAE_eV",
                         "native_eval_metric": p["eval_metric"],
                         "native_custom_metric_decimal_places": 6})
            if estimator.num_boosted_rounds() != chosen:
                raise RuntimeError("Saved XGBoost model does not match selected iterations")
    meta["wall_seconds"] = time.perf_counter()-timer
    meta["fit_seconds"] = meta["wall_seconds"]
    meta["modelparams"] = meta["native_params"]
    result = FitResult(estimator, meta, curve)
    if output_path is not None:
        meta["serialized"] = save_model(result, output_path)
    return result


def predict_head(result_or_estimator, family, task, X):
    model = result_or_estimator.estimator if isinstance(result_or_estimator, FitResult) else result_or_estimator
    X = np.asarray(X, dtype=np.float32, order="C")
    with threadpool_limits(limits=1, user_api="blas"):
        if family == "XGBoost":
            pred = model.inplace_predict(X)
        elif family == "RF" and task == "classifier":
            classes = list(model.classes_)
            pred = model.predict_proba(X)[:, classes.index(1)] if 1 in classes else np.zeros(len(X))
        elif family == "RF":
            pred = model.predict(X)
        else:
            raise ValueError(family)
    pred = _vector(pred, len(X), "prediction")
    if task == "classifier" and np.any((pred < 0) | (pred > 1)):
        raise ValueError("Classifier did not produce probabilities")
    return pred


def save_model(result, output_path):
    path = Path(output_path)
    expected = ".joblib" if result.metadata["family"] == "RF" else ".json"
    if path.suffix != expected:
        raise ValueError(f"Model path must end in {expected}")
    path.parent.mkdir(parents=True, exist_ok=True)
    if result.metadata["family"] == "RF":
        joblib.dump(result.estimator, path, compress=3)
    else:
        result.estimator.save_model(path)
    metadata_path = path.with_suffix(path.suffix+".metadata.json")
    curve_path = path.with_suffix(path.suffix+".curve.json")
    metadata_path.write_text(json.dumps(result.metadata, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    curve_path.write_text(json.dumps(result.curve, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    return {"model_path": str(path), "model_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "metadata_path": str(metadata_path), "metadata_sha256": hashlib.sha256(metadata_path.read_bytes()).hexdigest(),
            "curve_path": str(curve_path), "curve_sha256": hashlib.sha256(curve_path.read_bytes()).hexdigest()}


def load_model(path, family, expected_sha256=None):
    path = Path(path)
    if expected_sha256 is not None and hashlib.sha256(path.read_bytes()).hexdigest() != expected_sha256:
        raise ValueError("Saved model hash mismatch")
    if family == "RF":
        return joblib.load(path)
    if family == "XGBoost":
        model = xgb.Booster(params={"nthread": N_THREADS, "device": "cpu"})
        model.load_model(path)
        return model
    raise ValueError(family)
