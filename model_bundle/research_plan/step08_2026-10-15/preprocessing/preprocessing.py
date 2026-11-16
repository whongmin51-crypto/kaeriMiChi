"""Fit-only, fixed-schema preprocessing shared by tree and graph models.

No function in this module loads data or labels. Matrix rows supplied to ``fit``
must already be restricted to the explicit fitting identities.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping, Sequence

import numpy as np


VERSION = "anderson-step08-preprocessing-v1"


def identity_sha256(identities: Sequence[str]) -> str:
    """Hash the sorted identity set with an unambiguous JSON encoding."""
    payload = json.dumps(sorted(identities), ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _matrix(raw_values: Any, width: int) -> np.ndarray:
    values = np.asarray(raw_values, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != width:
        raise ValueError(f"Expected a 2D matrix with {width} columns, got {values.shape}.")
    if np.isinf(values).any():
        raise ValueError("Infinite feature values must be resolved in feature construction.")
    return values


def rows_to_matrix(rows: Sequence[Mapping[str, Any]], schema: Mapping[str, Any]) -> np.ndarray:
    """Expand sparse physical zeros, preserving explicit null as missing.

    ``schema['columns']`` is a name-to-definition mapping; a definition's default
    applies only when that key is absent, never when its value is explicitly null.
    """
    names = schema["ordered_feature_names"]
    columns = schema["columns"]
    result = np.full((len(rows), len(names)), np.nan, dtype=np.float64)
    for i, row in enumerate(rows):
        values = row["values"]
        for j, name in enumerate(names):
            value = values[name] if name in values else columns[name].get("default")
            if value is not None:
                result[i, j] = float(value)
    return _matrix(result, len(names))


def fit_preprocessor(raw_values: Any, feature_names: Sequence[str], fit_ids: Sequence[str]) -> dict:
    """Estimate medians and population scales from exactly the supplied fit rows."""
    names, ids = list(feature_names), list(fit_ids)
    if not names or len(names) != len(set(names)) or any(not isinstance(n, str) for n in names):
        raise ValueError("feature_names must be a nonempty list of unique strings.")
    if not ids or len(ids) != len(set(ids)) or any(not isinstance(i, str) for i in ids):
        raise ValueError("fit_ids must be a nonempty list of unique strings.")
    values = _matrix(raw_values, len(names))
    if values.shape[0] != len(ids):
        raise ValueError("fit_ids must align one-to-one with the fitting matrix rows.")
    columns = []
    for j, name in enumerate(names):
        col = values[:, j]
        observed = col[~np.isnan(col)]
        all_missing = len(observed) == 0
        median = 0.0 if all_missing else float(np.median(observed))
        imputed = np.where(np.isnan(col), median, col)
        mean = float(np.mean(imputed))
        raw_std = float(np.std(imputed, ddof=0))
        constant = all_missing or bool(np.max(observed) == np.min(observed))
        # Constants are determined from the actual observed range; rounding in a
        # mean/std calculation must never create a spurious active channel.
        std = 1.0 if constant or raw_std == 0.0 else raw_std
        if not all(np.isfinite(x) for x in (median, mean, raw_std, std)):
            raise ValueError(f"Preprocessing statistics overflowed for {name}.")
        columns.append({
            "name": name,
            "observed_count": int(len(observed)),
            "missing_count": int(len(ids) - len(observed)),
            "median": median,
            "mean": mean,
            "std": std,
            "imputed_population_std_before_constant_fallback": raw_std,
            "all_missing": all_missing,
            "constant": constant,
            "active": not constant,
            "learned_from_fit_only": True,
            "value_channel_policy": "fixed_zero" if constant else "impute_median",
            "model_indicator_policy": "fixed_one" if all_missing else "actual_missingness",
        })
    return {
        "version": VERSION,
        "feature_names": names,
        "model_feature_names": names + [name + "__missing" for name in names],
        "input_feature_count": len(names),
        "model_feature_count": 2 * len(names),
        "fit_ids": ids,
        "fit_id_sha256": identity_sha256(ids),
        "fit_row_count": len(ids),
        "statistics_ddof": 0,
        "statistics_use_imputed_training_values": True,
        "learned_from_fit_only": True,
        "columns": columns,
    }


def transform(raw_values: Any, state: Mapping[str, Any]) -> dict[str, np.ndarray]:
    """Transform without changing state; retain actual and unsupported-input masks."""
    if state.get("version") != VERSION:
        raise ValueError("Unsupported preprocessor version.")
    names = state["feature_names"]
    values = _matrix(raw_values, len(names))
    columns = state["columns"]
    if len(columns) != len(names) or [c["name"] for c in columns] != names:
        raise ValueError("State columns do not match the fixed feature order.")
    missing = np.isnan(values)
    median = np.array([c["median"] for c in columns], dtype=np.float64)
    mean = np.array([c["mean"] for c in columns], dtype=np.float64)
    std = np.array([c["std"] for c in columns], dtype=np.float64)
    constants = np.array([c["constant"] for c in columns], dtype=bool)
    all_missing = np.array([c["all_missing"] for c in columns], dtype=bool)
    imputed = np.where(missing, median, values)
    tree = imputed.copy()
    graph = (imputed - mean) / std
    # A value absent/constant during fitting has no learned response. Freeze its
    # numeric channel in all future datasets while exposing unsupported values.
    tree[:, constants] = 0.0
    graph[:, constants] = 0.0
    model_missing = missing.astype(np.float64)
    model_missing[:, all_missing] = 1.0
    result = {
        "tree_values": np.concatenate([tree, model_missing], axis=1),
        "graph_values": np.concatenate([graph, model_missing], axis=1),
        "missing_mask": missing.copy(),
        "unsupported_feature_mask": (~missing) & all_missing[None, :],
    }
    if not np.isfinite(result["tree_values"]).all() or not np.isfinite(result["graph_values"]).all():
        raise ValueError("Transformed features contain an infinite or NaN value.")
    return result
