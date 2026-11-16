"""Pure-array development metrics for the frozen Step07/08 protocol.

Each physical material pair has equal total weight within its source. Selection
metrics are computed within each source and then averaged across sources. No
reader, fitter, threshold optimizer or calibration is included in this module.
"""
from collections import Counter

import numpy as np


PROBABILITY_EPSILON = 1e-15


def _vector(values, name, length=None, *, finite=True):
    result = np.asarray(values, dtype=np.float64)
    if result.ndim != 1 or (length is not None and len(result) != length):
        raise ValueError(f'{name} must be a one-dimensional vector of matching length')
    if not len(result):
        raise ValueError(f'{name} cannot be empty')
    if finite and not np.all(np.isfinite(result)):
        raise ValueError(f'{name} must be finite')
    return result


def _identity_arrays(material_pair_ids, sources, length):
    pairs = np.asarray(material_pair_ids, dtype=str)
    if pairs.ndim != 1 or len(pairs) != length or np.any(pairs == ''):
        raise ValueError('material_pair_ids must contain a nonempty identity per row')
    datasets = np.full(length, 'all', dtype=str) if sources is None else np.asarray(sources, dtype=str)
    if datasets.ndim != 1 or len(datasets) != length or np.any(datasets == ''):
        raise ValueError('sources must contain a nonempty source per row')
    return pairs, datasets


def source_pair_weights(material_pair_ids, sources=None):
    """Normalized row weights: equal source, then pair, then pair configuration.

    A physical pair shared across sources receives its own allocation within each
    source, as required by the source-macro development comparison.
    """
    pairs = np.asarray(material_pair_ids, dtype=str)
    if pairs.ndim != 1 or not len(pairs):
        raise ValueError('At least one physical material pair identity is required')
    pairs, datasets = _identity_arrays(pairs, sources, len(pairs))
    counts = Counter(zip(datasets.tolist(), pairs.tolist()))
    distinct_sources = sorted(set(datasets.tolist()))
    source_pair_count = {
        source: len({pair for dataset, pair in counts if dataset == source})
        for source in distinct_sources
    }
    return np.asarray([
        1.0 / (len(distinct_sources) * source_pair_count[source] * counts[(source, pair)])
        for source, pair in zip(datasets, pairs)
    ], dtype=np.float64)


def _weights(weights, length):
    result = np.ones(length, dtype=np.float64) if weights is None else _vector(weights, 'weights', length)
    if np.any(result < 0) or not result.sum() > 0:
        raise ValueError('Weights must be nonnegative with positive total weight')
    return result / result.sum()


def weighted_average_precision(y_true, scores, weights=None):
    """Weighted non-interpolated AP with equal-score observations tied together.

    Return None when no positive weighted observations exist. All-positive AP is
    one. The score itself may be any finite ranking value.
    """
    y = _vector(y_true, 'y_true')
    if not np.all((y == 0) | (y == 1)):
        raise ValueError('Binary classification labels must be 0 or 1')
    score = _vector(scores, 'scores', len(y))
    w = _weights(weights, len(y))
    positive_weight = float(np.dot(w, y))
    if positive_weight == 0:
        return None
    nonzero = w > 0
    y, score, w = y[nonzero], score[nonzero], w[nonzero]
    order = np.argsort(-score, kind='stable')
    sorted_scores = score[order]
    cumulative_tp = np.cumsum(w[order] * y[order])
    cumulative_weight = np.cumsum(w[order])
    ends = np.r_[np.flatnonzero(sorted_scores[:-1] != sorted_scores[1:]), len(y) - 1]
    tp = cumulative_tp[ends]
    precision = tp / cumulative_weight[ends]
    recall_increment = np.diff(np.r_[0.0, tp]) / positive_weight
    return float(np.clip(np.dot(recall_increment, precision), 0.0, 1.0))


def weighted_auroc(y_true, scores, weights=None):
    """Weighted positive-vs-negative concordance; equal scores count one half."""
    y = _vector(y_true, 'y_true')
    if not np.all((y == 0) | (y == 1)):
        raise ValueError('Binary classification labels must be 0 or 1')
    score = _vector(scores, 'scores', len(y))
    w = _weights(weights, len(y))
    positive = float(np.dot(w, y))
    negative = float(np.dot(w, 1.0 - y))
    if positive == 0 or negative == 0:
        return None
    order = np.argsort(score, kind='stable')
    sorted_scores = score[order]
    starts = np.r_[0, np.flatnonzero(sorted_scores[:-1] != sorted_scores[1:]) + 1]
    positive_mass = np.add.reduceat(w[order] * y[order], starts)
    negative_mass = np.add.reduceat(w[order] * (1.0 - y[order]), starts)
    preceding_negative = np.cumsum(negative_mass) - negative_mass
    concordant = float(np.dot(positive_mass, preceding_negative + 0.5 * negative_mass))
    return float(np.clip(concordant / (positive * negative), 0.0, 1.0))


def _binary_metrics(y, probability, weights):
    w = _weights(weights, len(y))
    p = np.clip(probability, PROBABILITY_EPSILON, 1.0 - PROBABILITY_EPSILON)
    hard = probability >= 0.5  # Descriptive threshold, never the final route.
    tp = float(w[(y == 1) & hard].sum())
    fn = float(w[(y == 1) & ~hard].sum())
    fp = float(w[(y == 0) & hard].sum())
    tn = float(w[(y == 0) & ~hard].sum())
    precision = tp / (tp + fp) if tp + fp > 0 else None
    recall = tp / (tp + fn) if tp + fn > 0 else None
    specificity = tn / (tn + fp) if tn + fp > 0 else None
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn > 0 else None
    return {
        'average_precision': weighted_average_precision(y, probability, w),
        'log_loss': float(-np.dot(w, y * np.log(p) + (1.0 - y) * np.log1p(-p))),
        'auroc': weighted_auroc(y, probability, w),
        'brier': float(np.dot(w, (probability - y) ** 2)),
        'prevalence': float(np.dot(w, y)),
        'accuracy_at_0_5': tp + tn,
        'precision_at_0_5': precision,
        'failure_recall_at_0_5': recall,
        'specificity_at_0_5': specificity,
        'f1_at_0_5': f1,
        'balanced_accuracy_at_0_5': (recall + specificity) / 2 if recall is not None and specificity is not None else None,
        'confusion_weight_at_0_5': {'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp},
    }


def _macro(by_source, keys):
    values = {}
    support = {}
    for key in keys:
        available = [row[key] for row in by_source.values() if row[key] is not None]
        values[key] = float(np.mean(available)) if available else None
        support[key] = len(available)
    return values, support


def classification_metrics(y_true, p_fail, material_pair_ids, sources=None):
    """Source-macro classifier metrics for one fixed common admitted cohort.

    ``source_macro.average_precision`` is the frozen candidate selection score.
    No prediction is recalibrated. AP/AUROC source support is recorded explicitly.
    """
    y = _vector(y_true, 'y_true')
    if not np.all((y == 0) | (y == 1)):
        raise ValueError('Binary classification labels must be 0 or 1')
    probability = _vector(p_fail, 'p_fail', len(y))
    if np.any((probability < 0) | (probability > 1)):
        raise ValueError('p_fail must lie within [0,1]')
    pairs, datasets = _identity_arrays(material_pair_ids, sources, len(y))
    by_source = {}
    for source in sorted(set(datasets.tolist())):
        index = datasets == source
        row = _binary_metrics(y[index], probability[index], source_pair_weights(pairs[index]))
        row.update(n_records=int(index.sum()), n_pairs=len(set(pairs[index].tolist())),
                   n_failure_records=int(y[index].sum()))
        by_source[source] = row
    keys = ['average_precision', 'log_loss', 'auroc', 'brier', 'prevalence',
            'accuracy_at_0_5', 'precision_at_0_5', 'failure_recall_at_0_5',
            'specificity_at_0_5', 'f1_at_0_5', 'balanced_accuracy_at_0_5']
    macro, support = _macro(by_source, keys)
    return {
        'source_macro': macro, 'by_source': by_source,
        'source_macro_metric_support': support,
        'n_sources': len(by_source), 'n_records': len(y),
        'pooled_source_pair_weighted': _binary_metrics(y, probability, source_pair_weights(pairs, datasets)),
        'pooled_row_equal': _binary_metrics(y, probability, None),
        'selection_metric': 'source_macro.average_precision',
        'selection_direction': 'maximize',
        'hard_threshold_is_descriptive_only': True,
        'probability_calibration_performed': False,
    }


def reconstruct_gap(prediction, g_A=None, prediction_kind='direct_gap'):
    """Apply the common nonnegative PBE gap prediction semantics."""
    prediction = _vector(prediction, 'prediction')
    if prediction_kind == 'direct_gap':
        raw_gap = prediction
    elif prediction_kind == 'residual':
        if g_A is None:
            raise ValueError('Residual reconstruction requires the frozen Anderson gap g_A')
        baseline = _vector(g_A, 'g_A', len(prediction))
        if np.any(baseline < 0):
            raise ValueError('g_A must use the frozen nonnegative Anderson gap semantics')
        raw_gap = baseline + prediction
    else:
        raise ValueError('prediction_kind must be direct_gap or residual')
    return np.maximum(0.0, raw_gap)


def _regression_metrics(y, prediction, weights):
    w = _weights(weights, len(y))
    error = prediction - y
    mse = float(np.dot(w, error ** 2))
    target_mean = float(np.dot(w, y))
    variance = 0.0 if np.all(y == y[0]) else float(np.dot(w, (y - target_mean) ** 2))
    return {
        'mae_eV': float(np.dot(w, np.abs(error))),
        'rmse_eV': float(np.sqrt(mse)),
        'r2': float(1.0 - mse / variance) if variance > 0 else None,
        'bias_eV': float(np.dot(w, error)),
    }


def regression_metrics(g_PBE, prediction, material_pair_ids, sources=None,
                       g_A=None, prediction_kind='direct_gap'):
    """Score both regression heads in the same nonnegative PBE gap space.

    ``source_macro.mae_eV`` is the frozen candidate selection score. Source macro
    RMSE averages source RMSE values; it is distinct from the pooled weighted RMSE.
    R2 is unavailable for a source with constant true gaps, and support is shown.
    """
    target = _vector(g_PBE, 'g_PBE')
    if np.any(target < 0):
        raise ValueError('g_PBE must use the frozen nonnegative PBE gap semantics')
    pred = reconstruct_gap(prediction, g_A, prediction_kind)
    if len(pred) != len(target):
        raise ValueError('prediction and g_PBE lengths differ')
    pairs, datasets = _identity_arrays(material_pair_ids, sources, len(target))
    by_source = {}
    for source in sorted(set(datasets.tolist())):
        index = datasets == source
        row = _regression_metrics(target[index], pred[index], source_pair_weights(pairs[index]))
        row.update(n_records=int(index.sum()), n_pairs=len(set(pairs[index].tolist())))
        by_source[source] = row
    macro, support = _macro(by_source, ['mae_eV', 'rmse_eV', 'r2', 'bias_eV'])
    return {
        'source_macro': macro, 'by_source': by_source,
        'source_macro_metric_support': support,
        'n_sources': len(by_source), 'n_records': len(target),
        'pooled_source_pair_weighted': _regression_metrics(target, pred, source_pair_weights(pairs, datasets)),
        'pooled_row_equal': _regression_metrics(target, pred, None),
        'prediction_kind': prediction_kind,
        'projection': 'max(0,direct_prediction) or max(0,g_A+residual_prediction)',
        'selection_metric': 'source_macro.mae_eV',
        'selection_direction': 'minimize',
    }


def early_stop_score(task, y_true, prediction, material_pair_ids, sources=None, g_A=None):
    """Cheap inner-stop scalar: macro weighted BCE or reconstructed gap MAE.

    For either regression task, y_true is always the nonnegative *PBE gap*, even
    when prediction is the residual head output. Iteration selection stays inside
    the inner-fit/inner-stop split and does not use an outer validation score.
    """
    target = _vector(y_true, 'y_true')
    pred = _vector(prediction, 'prediction', len(target))
    pairs, datasets = _identity_arrays(material_pair_ids, sources, len(target))
    w = source_pair_weights(pairs, datasets)
    if task == 'classifier':
        if not np.all((target == 0) | (target == 1)) or np.any((pred < 0) | (pred > 1)):
            raise ValueError('Binary labels and probabilities within [0,1] are required')
        p = np.clip(pred, PROBABILITY_EPSILON, 1.0 - PROBABILITY_EPSILON)
        return float(-np.dot(w, target * np.log(p) + (1.0 - target) * np.log1p(-p)))
    kinds = {'residual_regression': 'residual', 'direct_gap_regression': 'direct_gap'}
    if task not in kinds:
        raise ValueError('Unknown Step07 task')
    if np.any(target < 0):
        raise ValueError('Regression inner-stop target must be the nonnegative PBE gap')
    gap = reconstruct_gap(pred, g_A, kinds[task])
    return float(np.dot(w, np.abs(gap - target)))
