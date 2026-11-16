"""Deterministic CPU training, inner-stop selection and restartable checkpoints.

This module does not launch a scientific fit when imported. The root Step10
orchestrator is the sole caller of fit_model and owns the scientific run lock.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import time
from datetime import datetime, timezone

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from graph_data import CoreGraphData, collate_pairs, pack_graphs, to_device, identity_sha256, sha256


ROOT = Path(__file__).resolve().parent
TASK_NAMES = ("classifier", "residual", "direct_gap")
TARGET_INDEX = {"classifier": 0, "residual": 1, "direct_gap": 2}


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


METRICS = _load_module("step09_pure_array_metrics", ROOT.parent / "step09_2026-10-15/metrics.py")


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(temporary, path)


def atomic_torch(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def peak_memory_mib():
    try:
        import resource
        return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
    except (ImportError, AttributeError):
        return None


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(4)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    torch.use_deterministic_algorithms(True)


def task_names(task):
    if task == "multi_task":
        return TASK_NAMES
    if task not in TASK_NAMES:
        raise ValueError(f"Unknown task {task}")
    return (task,)


def _targets(data, ids):
    rows = data.metadata_for(ids)
    return {"classifier": np.asarray([r["y"] for r in rows], dtype=float),
            "residual": np.asarray([r["r_eV"] for r in rows], dtype=float),
            "direct_gap": np.asarray([r["g_PBE_eV"] for r in rows], dtype=float),
            "g_A": np.asarray([r["g_A_eV"] for r in rows], dtype=float),
            "pairs": np.asarray([r["material_pair_id"] for r in rows], dtype=str),
            "sources": np.asarray([r["dataset"] for r in rows], dtype=str)}


def fit_loss_scales(data, fit_ids):
    target = _targets(data, fit_ids)
    weights = METRICS.source_pair_weights(target["pairs"], target["sources"])
    return {"residual_std_eV": max(float(np.std(target["residual"], ddof=0)), 1e-3),
            "direct_gap_std_eV": max(float(np.std(target["direct_gap"], ddof=0)), 1e-3),
            "anderson_source_pair_mae_eV": max(float(np.dot(weights, np.abs(
                target["g_A"] - target["direct_gap"]))), 1e-6),
            "fit_ids_sha256": identity_sha256(fit_ids),
            "estimated_from_actual_fit_only": True}


def loss_terms(outputs, targets, tasks, scales, weights):
    raw = {}
    for name in tasks:
        truth = targets[:, TARGET_INDEX[name]]
        if name == "classifier":
            raw[name] = nn.functional.binary_cross_entropy_with_logits(outputs[name], truth)
        else:
            raw[name] = nn.functional.mse_loss(outputs[name], truth)
    if len(tasks) == 1:
        return next(iter(raw.values())), raw
    total = weights[0] * raw["classifier"]
    total = total + weights[1] * raw["residual"] / scales["residual_std_eV"] ** 2
    total = total + weights[2] * raw["direct_gap"] / scales["direct_gap_std_eV"] ** 2
    return total, raw


def _model(family, params, tasks):
    from models import build_model
    return build_model(family, params, n_numeric=1540, tasks=tasks)


def _sigmoid(values):
    values = np.asarray(values, dtype=np.float64)
    result = np.empty_like(values)
    positive = values >= 0
    result[positive] = 1 / (1 + np.exp(-values[positive]))
    exponential = np.exp(values[~positive])
    result[~positive] = exponential / (1 + exponential)
    return result


@torch.no_grad()
def predict_arrays(model, data, ids, scope, tasks, batch_size=256):
    """Predict in ID order; optional unique-graph eval caching changes no math."""
    model.eval()
    ids = list(ids)
    data.validate_ids(ids)
    output = {name: [] for name in tasks}
    if hasattr(model, "encode_graph") and hasattr(model, "fuse_embeddings"):
        # BN is frozen in eval; isolated crystal encodings can be reused exactly.
        graph_ids = sorted({g for i in ids for g in data.graph_pairs[i].values()})
        graph_lookup = {g: j for j, g in enumerate(graph_ids)}
        graphs = pack_graphs([(g, data.graph_cache[g]) for g in graph_ids])
        embedding = model.encode_graph(graphs)
        numeric = data.numeric(scope)
        for start in range(0, len(ids), batch_size):
            batch_ids = ids[start:start + batch_size]
            lower_index = torch.tensor([graph_lookup[data.graph_pairs[i]["lower"]] for i in batch_ids])
            upper_index = torch.tensor([graph_lookup[data.graph_pairs[i]["upper"]] for i in batch_ids])
            numeric_index = torch.tensor([data.index[i] for i in batch_ids])
            result = model.fuse_embeddings(embedding[lower_index], embedding[upper_index], numeric[numeric_index])
            for name in tasks:
                output[name].append(result[name].detach().cpu().numpy().astype(np.float64))
    else:
        loader = DataLoader(data.dataset(scope, ids), batch_size=batch_size, shuffle=False,
                            num_workers=0, collate_fn=collate_pairs)
        for batch in loader:
            result = model(batch)
            for name in tasks:
                output[name].append(result[name].detach().cpu().numpy().astype(np.float64))
    output = {name: np.concatenate(parts) for name, parts in output.items()}
    if "classifier" in output:
        output["classifier"] = _sigmoid(output["classifier"])
    if any(not np.isfinite(values).all() for values in output.values()):
        raise FloatingPointError("Nonfinite model predictions")
    return output


def early_stop_metrics(data, ids, predictions, tasks, scales):
    target = _targets(data, ids)
    scores = {}
    for name in tasks:
        if name == "classifier":
            score = METRICS.early_stop_score("classifier", target[name], predictions[name],
                                            target["pairs"], target["sources"])
        else:
            task_name = "residual_regression" if name == "residual" else "direct_gap_regression"
            score = METRICS.early_stop_score(task_name, target["direct_gap"], predictions[name],
                                            target["pairs"], target["sources"], target["g_A"])
        scores[name] = float(score)
    if len(tasks) == 1:
        joint = next(iter(scores.values()))
    else:
        joint = (scores["classifier"] / math.log(2) + scores["residual"] /
                 scales["anderson_source_pair_mae_eV"] + scores["direct_gap"] /
                 scales["anderson_source_pair_mae_eV"]) / 3
    return float(joint), scores


def _validate_scope(data, scope, fit_ids, stop_ids):
    state = data.state(scope)
    if set(state["fit_ids"]) != set(fit_ids) or len(state["fit_ids"]) != len(fit_ids):
        raise ValueError("Actual fitting IDs must exactly match the frozen preprocessing scope")
    if stop_ids is not None:
        data.validate_ids(stop_ids)
        if set(fit_ids) & set(stop_ids):
            raise ValueError("Inner-fit and inner-stop identities overlap")
        if not (scope.startswith("fold") and scope.endswith("_inner_fit")):
            raise ValueError("Early stopping is only legal in the frozen inner-fit scope")
        fold = int(scope[4])
        if set(stop_ids) != set(data.subset_ids("inner_stop", fold)):
            raise ValueError("Early-stop identities must equal the frozen inner-stop pool")


def fit_model(data, family, params, task, fit_ids, scope, run_dir, seed,
              stop_ids=None, epochs=300, patience=30, loss_weights=None, resume=True):
    """Fit one scientific stage; no outer-validation/calibration labels are used.

    Inner stage: supply stop_ids to choose best epoch. Outer/final stage: fresh
    run directory and stop_ids=None, epochs=frozen chosen iteration count.
    A completed matching stage is idempotent. An interrupted stage resumes its
    optimizer, random states, epoch and best-state metadata from last.pt.
    """
    started = time.perf_counter()
    tasks = task_names(task)
    fit_ids = list(fit_ids)
    stop_ids = None if stop_ids is None else list(stop_ids)
    data.validate_ids(fit_ids)
    _validate_scope(data, scope, fit_ids, stop_ids)
    epochs, patience = int(epochs), int(patience)
    if not 1 <= epochs <= 300 or patience < 1:
        raise ValueError("Scientific epoch cap is 300 and patience must be positive")
    weights = [1.0, 1.0, 1.0] if loss_weights is None else list(map(float, loss_weights))
    if len(weights) != 3 or min(weights) <= 0:
        raise ValueError("Multi-task weights must be three positive registered values")
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    set_seed(seed)
    state = data.state(scope)
    scales = fit_loss_scales(data, fit_ids)
    specification = {"family": family, "params": params, "task": task, "tasks": list(tasks),
                     "seed": int(seed), "fit_scope": scope,
                     "fit_id_sha256": identity_sha256(fit_ids), "n_fit": len(fit_ids),
                     "stop_id_sha256": None if stop_ids is None else identity_sha256(stop_ids),
                     "n_stop": 0 if stop_ids is None else len(stop_ids), "epochs_cap": epochs,
                     "patience": patience, "loss_weights": weights,
                     "fit_loss_scales": scales,
                     "preprocessor_sha256": data.states[scope]["sha256"],
                     "input_binding": data.input_binding,
                     "optimizer": "torch.optim.AdamW", "training_row_weighting": "row_equal",
                     "fit_target_scaling": "single-task raw eV; multi-task fit-only variance loss normalization",
                     "torch_version": str(torch.__version__), "numpy_version": str(np.__version__),
                     "device": "cpu", "torch_threads": 4, "interop_threads": 1,
                     "models_sha256": sha256(ROOT / "models.py"),
                     "engine_sha256": sha256(Path(__file__)),
                     "data_reader_sha256": sha256(ROOT / "graph_data.py")}
    spec_sha = hashlib.sha256(json.dumps(specification, sort_keys=True, ensure_ascii=False,
                                        separators=(",", ":")).encode()).hexdigest()
    existing_spec = run_dir / "fit_specification.json"
    if existing_spec.exists():
        prior = json.loads(existing_spec.read_text(encoding="utf-8"))
        if prior.get("specification_sha256") != spec_sha:
            raise ValueError("Resume specification changed; use a new registered run identity")
    else:
        atomic_json(existing_spec, {**specification, "specification_sha256": spec_sha,
                                   "created_utc": utc_now(), "fit_ids": fit_ids,
                                   "stop_ids": stop_ids})
    completed_path = run_dir / "summary.json"
    if completed_path.exists():
        summary = json.loads(completed_path.read_text(encoding="utf-8"))
        if summary.get("status") == "PASS" and summary.get("specification_sha256") == spec_sha:
            return summary
    model = _model(family, params, tasks)
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=float(params.get("learning_rate", params.get("lr", 1e-3))),
                                  weight_decay=float(params.get("weight_decay", 0)))
    batch_size = int(params.get("batch_size", 64))
    if batch_size < 2:
        raise ValueError("Batch size must be at least 2")
    best_epoch, last_epoch, bad_epochs = 0, 0, 0
    best_score = float("inf")
    restored_elapsed = 0.0
    if resume and (run_dir / "last.pt").exists():
        checkpoint = torch.load(run_dir / "last.pt", map_location="cpu", weights_only=False)
        if checkpoint["specification_sha256"] != spec_sha:
            raise ValueError("Checkpoint specification differs from requested run")
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        best_epoch, last_epoch, bad_epochs = checkpoint["best_epoch"], checkpoint["epoch"], checkpoint["bad_epochs"]
        best_score = checkpoint["best_score"]
        random.setstate(checkpoint["random_state"])
        np.random.set_state(checkpoint["numpy_random_state"])
        torch.set_rng_state(checkpoint["torch_rng_state"])
        restored_elapsed = checkpoint.get("elapsed_training_seconds", 0.0)
        # Discard any curve line written after the last committed checkpoint.
        curve_path = run_dir / "curve.jsonl"
        if curve_path.exists():
            rows = [json.loads(line) for line in curve_path.read_text(encoding="utf-8").splitlines() if line]
            curve_path.write_text("".join(json.dumps(row, allow_nan=False) + "\n" for row in rows
                                            if row["epoch"] <= last_epoch), encoding="utf-8")
    elif (run_dir / "curve.jsonl").exists():
        # The first epoch may have written its curve/best file immediately
        # before a power loss. Without last.pt no epoch is committed: replay
        # deterministically from the original seed, discarding that curve.
        (run_dir / "curve.jsonl").write_text("", encoding="utf-8")
    parameter_count = sum(p.numel() for p in model.parameters())
    for epoch in range(last_epoch + 1, epochs + 1):
        if stop_ids is not None and bad_epochs >= patience:
            break
        epoch_started = time.perf_counter()
        generator = torch.Generator().manual_seed(int(seed) + epoch)
        loader = DataLoader(data.dataset(scope, fit_ids), batch_size=batch_size, shuffle=True,
                            num_workers=0, generator=generator, collate_fn=collate_pairs,
                            drop_last=False)
        model.train()
        total_loss = 0.0
        term_totals = {name: 0.0 for name in tasks}
        count = 0
        for batch in loader:
            optimizer.zero_grad(set_to_none=True)
            outputs = model(batch)
            loss, terms = loss_terms(outputs, batch["targets"], tasks, scales, weights)
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("Nonfinite scientific training loss")
            loss.backward()
            if any(p.grad is not None and not bool(torch.isfinite(p.grad).all()) for p in model.parameters()):
                raise FloatingPointError("Nonfinite scientific gradient")
            optimizer.step()
            n = len(batch["identities"])
            count += n
            total_loss += float(loss.detach()) * n
            for name, value in terms.items():
                term_totals[name] += float(value.detach()) * n
        stop_score, per_task_score = None, None
        if stop_ids is not None:
            predictions = predict_arrays(model, data, stop_ids, scope, tasks)
            stop_score, per_task_score = early_stop_metrics(data, stop_ids, predictions, tasks, scales)
            improved = stop_score < best_score
            if improved:
                best_epoch, best_score, bad_epochs = epoch, stop_score, 0
            else:
                bad_epochs += 1
        else:
            improved = True
            best_epoch, best_score, bad_epochs = epoch, 0.0, 0
        last_epoch = epoch
        row = {"epoch": epoch, "train_total_loss": total_loss / count,
               "train_raw_task_losses": {name: value / count for name, value in term_totals.items()},
               "inner_stop_score": stop_score, "inner_stop_task_scores": per_task_score,
               "best_epoch_so_far": best_epoch, "bad_epochs": bad_epochs,
               "n_fit_seen": count, "epoch_seconds": time.perf_counter() - epoch_started,
               "peak_process_memory_mib": peak_memory_mib(), "utc": utc_now()}
        with (run_dir / "curve.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
        checkpoint = {"model_state_dict": model.state_dict(),
                      "optimizer_state_dict": optimizer.state_dict(), "epoch": epoch,
                      "best_epoch": best_epoch, "best_score": best_score, "bad_epochs": bad_epochs,
                      "specification_sha256": spec_sha, "specification": specification,
                      "random_state": random.getstate(), "numpy_random_state": np.random.get_state(),
                      "torch_rng_state": torch.get_rng_state(),
                      "elapsed_training_seconds": restored_elapsed + time.perf_counter() - started,
                      "parameter_count": parameter_count, "created_utc": utc_now()}
        if improved:
            # Write the best state first; last.pt is the epoch commit point.
            # If interrupted between these writes, replay from the prior last
            # checkpoint reproduces this epoch and its best-state replacement.
            atomic_torch(run_dir / "best.pt", checkpoint)
        atomic_torch(run_dir / "last.pt", checkpoint)
    summary = {"status": "PASS", "completed": True, "family": family, "task": task,
               "best_epoch": best_epoch, "last_epoch": last_epoch,
               "best_inner_stop_score": None if stop_ids is None else best_score,
               "n_fit": len(fit_ids), "n_stop": 0 if stop_ids is None else len(stop_ids),
               "fit_id_sha256": identity_sha256(fit_ids), "fit_scope": scope,
               "specification_sha256": spec_sha,
               "specification_path": str(existing_spec), "checkpoint_path": str(run_dir / "best.pt"),
               "checkpoint_sha256": sha256(run_dir / "best.pt"),
               "last_checkpoint_sha256": sha256(run_dir / "last.pt"),
               "curve_sha256": sha256(run_dir / "curve.jsonl"),
               "parameter_count": parameter_count,
               "training_seconds": restored_elapsed + time.perf_counter() - started,
               "peak_process_memory_mib": peak_memory_mib(), "completed_utc": utc_now(),
               "outer_validation_used_for_epoch_selection": False,
               "calibration_external_test_labels_read": False}
    atomic_json(completed_path, summary)
    return summary


def predict_model(data, run_dir, ids, scope=None, checkpoint="best.pt"):
    run_dir = Path(run_dir)
    saved = torch.load(run_dir / checkpoint, map_location="cpu", weights_only=False)
    specification = saved["specification"]
    if scope is None:
        scope = specification["fit_scope"]
    if scope != specification["fit_scope"]:
        raise ValueError("Prediction must use the state bound to the fitted model")
    if sha256(ROOT / "models.py") != specification["models_sha256"]:
        raise ValueError("Model source differs from the frozen fitted checkpoint")
    model = _model(specification["family"], specification["params"], tuple(specification["tasks"]))
    model.load_state_dict(saved["model_state_dict"])
    result = predict_arrays(model, data, ids, scope, tuple(specification["tasks"]))
    result["ids"] = list(ids)
    return result
