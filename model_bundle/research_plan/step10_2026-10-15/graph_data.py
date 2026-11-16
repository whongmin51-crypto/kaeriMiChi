"""Frozen C1 numeric features and isolated monolayer graphs for Step10.

Only the already audited Step09 fit_core metadata contains labels. This reader
never opens a calibration, external-validation, or final-test label file.
"""
from __future__ import annotations

from functools import lru_cache
import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import Dataset


ROOT = Path(__file__).resolve().parent
PLAN = ROOT.parent
TASKS = ("classifier", "residual", "direct_gap")


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def identity_sha256(ids):
    return hashlib.sha256(json.dumps(sorted(ids), ensure_ascii=False,
                                    separators=(",", ":")).encode()).hexdigest()


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_preprocessing = _load_module("step08_preprocessing", PLAN / "step08_2026-10-15/preprocessing/preprocessing.py")


class CoreGraphData:
    """Read fixed-core rows once, then transform with the exact fit-scope state."""

    def __init__(self, root=ROOT, verify_hashes=True):
        self.root = Path(root)
        self.plan = self.root.parent
        self.s7 = self.plan / "step07_2026-10-15"
        self.s8 = self.plan / "step08_2026-10-15"
        self.s9 = self.plan / "step09_2026-10-15"
        partition_path = self.s7 / "record_partitions.jsonl"
        self.partition = {}
        with partition_path.open(encoding="utf-8") as stream:
            for line in stream:
                row = json.loads(line)
                if row["partition"] == "fit_core":
                    self.partition[row["heterostructure_id"]] = row
        metadata_path = self.s9 / "data_cache/core_metadata.json"
        # Step09's immutable execution manifest supplies the expected digest;
        # recording only the current digest would not bind the label source.
        frozen_s9 = json.loads((self.s9 / "code_manifest.json").read_text(encoding="utf-8"))
        for relative in ("data_cache/core_metadata.json", "data_cache/manifest.json"):
            if sha256(self.s9 / relative) != frozen_s9["files"][relative]:
                raise ValueError(f"Step09 frozen core-cache SHA mismatch: {relative}")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if not isinstance(metadata, list):
            raise ValueError("Step09 core metadata must be a list")
        self.metadata = {r["heterostructure_id"]: r for r in metadata}
        if len(self.metadata) != len(metadata) or set(self.metadata) != set(self.partition):
            raise ValueError("Core label cache identities differ from fixed Step07 fit_core")
        self.ids = list(self.metadata)
        frozen_cache = json.loads((self.s9 / "data_cache/manifest.json").read_text(encoding="utf-8"))
        if "core_ids" in frozen_cache and frozen_cache["core_ids"] != self.ids:
            raise ValueError("Step09 frozen core-cache row order differs from label metadata")
        s8_validation = json.loads((self.s8 / "validation.json").read_text(encoding="utf-8"))
        expected_s8 = {str(path).replace("\\", "/").split("/step08_2026-10-15/", 1)[-1]: digest
                       for path, digest in s8_validation["output_sha256"].items()}
        for relative in ("features/raw_feature_matrix.npz", "features/feature_schema.json",
                         "features/features.jsonl", "preprocessing/state_manifest.json",
                         "preprocessing/preprocessing.py", "graphs/cache_manifest.jsonl"):
            if sha256(self.s8 / relative) != expected_s8[relative]:
                raise ValueError(f"Step08 frozen graph/numeric input SHA mismatch: {relative}")
        for relative, key in (("cv_folds.json", "step07_cv"),
                              ("record_partitions.jsonl", "step07_partitions")):
            if sha256(self.s7 / relative) != frozen_cache["input_hashes"][key]:
                raise ValueError(f"Step07 identities differ from the audited Step09 cache: {relative}")
        self.index = {identity: i for i, identity in enumerate(self.ids)}
        self.folds = json.loads((self.s7 / "cv_folds.json").read_text(encoding="utf-8"))["folds"]
        archive = self.s8 / "features/raw_feature_matrix.npz"
        with np.load(archive, allow_pickle=False) as data:
            source_ids = data["record_ids"].astype(str).tolist()
            row_index = {identity: i for i, identity in enumerate(source_ids)}
            # Unlabelled rows outside core are never retained in a fitting array.
            self.raw = data["values"][[row_index[i] for i in self.ids]].copy()
            self.feature_names = data["feature_names"].astype(str).tolist()
        self.graph_pairs = {}
        with (self.s8 / "features/features.jsonl").open(encoding="utf-8") as stream:
            for line in stream:
                row = json.loads(line)
                identity = row["heterostructure_id"]
                if identity not in self.index:
                    continue
                if row["profile"] != "available_initial_stacking":
                    raise ValueError("Only frozen C1 features may enter Step10")
                self.graph_pairs[identity] = row["graph_ids"]
        if set(self.graph_pairs) != set(self.index):
            raise ValueError("Missing fixed-core graph pair")
        self.graph_manifest = {}
        with (self.s8 / "graphs/cache_manifest.jsonl").open(encoding="utf-8") as stream:
            for line in stream:
                row = json.loads(line)
                self.graph_manifest[row["graph_id"]] = row
        self.states = {r["scope"]: r for r in json.loads(
            (self.s8 / "preprocessing/state_manifest.json").read_text(encoding="utf-8"))["states"]}
        self._state_cache = {}
        self._numeric_cache = {}
        self.graph_cache = {}
        required = {g for pair in self.graph_pairs.values() for g in pair.values()}
        for graph_id in sorted(required):
            row = self.graph_manifest[graph_id]
            path = self.s8 / "graphs/cache" / (graph_id.split(":", 1)[1] + ".npz")
            if verify_hashes and sha256(path) != row["sha256"]:
                raise ValueError(f"Graph SHA mismatch: {graph_id}")
            with np.load(path, allow_pickle=False) as data:
                graph = {
                    "atomic_numbers": torch.from_numpy(data["atomic_numbers"].astype(np.int64)),
                    "edge_index": torch.from_numpy(data["edge_index"].astype(np.int64)),
                    "edge_rbf": torch.from_numpy(data["edge_rbf"].astype(np.float32)),
                    "edge_vectors_A": torch.from_numpy(data["edge_vectors_A"].astype(np.float32)),
                }
            if graph["edge_index"].numel() == 0 or graph["atomic_numbers"].numel() == 0:
                raise ValueError("Empty crystal graphs are not admitted")
            self.graph_cache[graph_id] = graph
        self.input_binding = {
            "core_metadata_sha256": sha256(metadata_path),
            "core_cache_manifest_sha256": sha256(self.s9 / "data_cache/manifest.json"),
            "fixed_partitions_sha256": sha256(partition_path),
            "cv_folds_sha256": sha256(self.s7 / "cv_folds.json"),
            "raw_features_sha256": sha256(archive),
            "feature_schema_sha256": sha256(self.s8 / "features/feature_schema.json"),
            "graph_manifest_sha256": sha256(self.s8 / "graphs/cache_manifest.jsonl"),
            "preprocessing_manifest_sha256": sha256(self.s8 / "preprocessing/state_manifest.json"),
            "profile": "available_initial_stacking",
            "core_id_sha256": identity_sha256(self.ids),
            "core_count": len(self.ids),
            "unique_core_graph_count": len(self.graph_cache),
            "labels_source": "Step09 audited core_metadata.json only",
            "calibration_external_test_labels_read": False,
        }

    def subset_ids(self, subset="fit_core", fold=None):
        if fold is None:
            if subset != "fit_core":
                raise ValueError("Only fit_core is available without fold")
            return list(self.ids)
        if type(fold) is not int or fold not in (0, 1, 2):
            raise ValueError("fold must be 0, 1, or 2")
        if subset not in ("outer_train", "outer_validation", "inner_fit", "inner_stop"):
            raise ValueError("Unknown legal development subset")
        row = next(r for r in self.folds if r["fold"] == fold)
        ids = row[subset + "_ids"]
        self.validate_ids(ids)
        return list(ids)

    def validate_ids(self, ids):
        if not ids or len(ids) != len(set(ids)) or not set(ids) <= set(self.index):
            raise ValueError("Every requested identity must be unique and in fixed fit_core")

    def state(self, scope):
        if scope not in self.states:
            raise ValueError("Preprocessing scope is not one of the frozen seven states")
        if scope not in self._state_cache:
            entry = self.states[scope]
            path = self.s8 / entry["path"]
            if sha256(path) != entry["sha256"]:
                raise ValueError("Frozen preprocessing hash mismatch")
            state = json.loads(path.read_text(encoding="utf-8"))
            if state["feature_names"] != self.feature_names:
                raise ValueError("Frozen feature order mismatch")
            self._state_cache[scope] = state
        return self._state_cache[scope]

    def numeric(self, scope):
        if scope not in self._numeric_cache:
            transformed = _preprocessing.transform(self.raw, self.state(scope))
            self._numeric_cache[scope] = torch.from_numpy(transformed["graph_values"].astype(np.float32))
        return self._numeric_cache[scope]

    def dataset(self, scope, ids=None, *, subset="fit_core", fold=None):
        ids = self.subset_ids(subset, fold) if ids is None else list(ids)
        self.validate_ids(ids)
        return PairDataset(self, scope, ids)

    def metadata_for(self, ids):
        self.validate_ids(ids)
        return [self.metadata[i] for i in ids]


class PairDataset(Dataset):
    def __init__(self, data, scope, ids):
        self.data, self.scope, self.ids = data, scope, list(ids)
        self.numeric = data.numeric(scope)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        identity = self.ids[index]
        meta = self.data.metadata[identity]
        pair = self.data.graph_pairs[identity]
        return {"identity": identity,
                "numeric": self.numeric[self.data.index[identity]],
                "lower": (pair["lower"], self.data.graph_cache[pair["lower"]]),
                "upper": (pair["upper"], self.data.graph_cache[pair["upper"]]),
                "targets": torch.tensor([meta["y"], meta["r_eV"], meta["g_PBE_eV"]], dtype=torch.float32),
                "g_A": float(meta["g_A_eV"])}


def pack_graphs(graph_items):
    """Pack real edges only; a repeated graph is encoded once per layer/batch."""
    unique, sample_index, seen = [], [], {}
    for graph_id, graph in graph_items:
        if graph_id not in seen:
            seen[graph_id] = len(unique)
            unique.append(graph)
        sample_index.append(seen[graph_id])
    numbers, edges, radial, vectors, node_batch = [], [], [], [], []
    offset = 0
    for index, graph in enumerate(unique):
        n = len(graph["atomic_numbers"])
        numbers.append(graph["atomic_numbers"])
        edges.append(graph["edge_index"] + offset)
        radial.append(graph["edge_rbf"])
        vectors.append(graph["edge_vectors_A"])
        node_batch.append(torch.full((n,), index, dtype=torch.long))
        offset += n
    return {"atomic_numbers": torch.cat(numbers), "edge_index": torch.cat(edges, dim=1),
            "edge_rbf": torch.cat(radial), "edge_vectors_A": torch.cat(vectors),
            "node_graph_index": torch.cat(node_batch), "n_graphs": len(unique),
            "sample_index": torch.tensor(sample_index, dtype=torch.long),
            "sample_count": len(graph_items)}


def collate_pairs(rows):
    return {"numeric": torch.stack([r["numeric"] for r in rows]),
            "lower": pack_graphs([r["lower"] for r in rows]),
            "upper": pack_graphs([r["upper"] for r in rows]),
            "targets": torch.stack([r["targets"] for r in rows]),
            "g_A": torch.tensor([r["g_A"] for r in rows], dtype=torch.float32),
            "identities": [r["identity"] for r in rows]}


def to_device(batch, device):
    if torch.is_tensor(batch):
        return batch.to(device)
    if isinstance(batch, dict):
        return {key: to_device(value, device) for key, value in batch.items()}
    return batch
