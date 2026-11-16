"""Frozen C1 prediction for arbitrary model-ready inputs, without target access.

Run in the original Step10 WSL environment. The contract binds the selected
five-seed heads, fit_core preprocessing, calibration, route and interval rules.
It accepts a new record ID; it never opens feature rows or labels by ID.
"""
from __future__ import annotations

import os
for _key in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "BLIS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "1"
os.environ["OMP_NUM_THREADS"] = "4"

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
RP = ROOT.parent
S8 = RP / "step08_2026-10-15"
S10 = RP / "step10_2026-10-15"
VERSION = "C1-step11-freeze-2026-10-25-v1"
CONTRACT = ROOT / "prediction_contract.json"
PROTOCOL = {
    "exchange_correlation": "PBE", "soc": False,
    "gap_channel": "nonnegative_operational_PBE_gap",
    "contact_reference": "contact_side_vacuum",
    "feature_profile": "available_initial_stacking", "geometry_stage": "initial_candidate",
}
GRAPH_FORMAT = "step08_2d_radius5_shell24_rbf32"
CRITICAL = [f"{position}__{name}" for position in ("lower", "upper")
            for name in ("gap_eV", "vbm_contact_eV", "cbm_contact_eV", "contact_side_canonical_code")]


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def local_path(value, base=ROOT):
    value = str(value)
    if len(value) > 2 and value[1] == ":":
        value = "/mnt/" + value[0].lower() + "/" + value[2:].replace("\\", "/").lstrip("/")
    path = Path(value)
    return path if path.is_absolute() else Path(base) / path


def load_preprocessing():
    spec = importlib.util.spec_from_file_location("step11_frozen_preprocessing", S8 / "preprocessing/preprocessing.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def freeze_contract():
    """Bind already-frozen outputs; no fitting, new selection or label access."""
    paths = [ROOT / "base_selection_freeze.json", ROOT / "calibration/probability.json",
             ROOT / "calibration/decision.json", ROOT / "calibration/interval.json", ROOT / "upstream_bindings.json",
             S8 / "preprocessing/states/fit_core.json", S8 / "features/feature_schema.json",
             S8 / "preprocessing/preprocessing.py", S10 / "models.py", S10 / "graph_engine.py", S10 / "graph_data.py"]
    base, probability, decision, interval, upstream = [read(path) for path in paths[:5]]
    assert base["status"] == "FROZEN_BEFORE_CALIBRATION"
    assert probability["status"] == decision["status"] == interval["status"] == "FROZEN"
    assert probability["base_selection_sha256"] == decision["base_selection_sha256"] == sha(paths[0])
    assert decision["probability_sha256"] == interval["probability_sha256"] == sha(paths[1])
    assert interval["decision_sha256"] == sha(paths[2])
    assert upstream["status"] == "PASS"
    manifest = read(S10 / "final_models_manifest.json")
    assert manifest["status"] == "PASS" and manifest["model_count"] == 40
    selected = {}
    for task, selection in base["heads"].items():
        entries = [e for e in manifest["models"] if e["model"] + "_" + e["task"] == selection["candidate"]]
        assert sorted(e["seed"] for e in entries) == list(range(20261015, 20261020))
        selected[task] = sorted(entries, key=lambda e: e["seed"])
        for entry in entries:
            path = S10 / entry["path"]
            assert sha(path) == entry["sha256"]
            assert upstream["files_sha256"][str(path.relative_to(RP))] == entry["sha256"]
    result = {"status": "FROZEN", "model_version": VERSION,
              "bindings_sha256": {str(path.relative_to(RP)): sha(path) for path in paths},
              "selected_models": selected, "service": base["service"],
              "protocol_required": PROTOCOL, "reference_tracks": ["native_free", "matched_strain"],
              "graph_format": GRAPH_FORMAT, "required_feature_names": read(paths[6])["ordered_feature_names"],
              "critical_finite_feature_names": CRITICAL + ["anderson_signed_eV", "anderson_nonnegative_eV"],
              "input_policy": "all 770 feature names present; optional nulls retain original imputation and missing indicators",
              "support_policy": "unseen elements, newly finite training-all-missing values, or true-twist input abstain",
              "unsupported_protocol": "invalid_input; no numerical prediction",
              "source_and_track_claim": "input support checks only; held-out predictive performance is evaluated in Step12",
              "label_access": False}
    if CONTRACT.exists():
        assert read(CONTRACT) == result, "Frozen prediction contract changed"
    else:
        save(CONTRACT, result)
    return result


class InputAdapter:
    def __init__(self, numeric, graphs, graph_hashes):
        self.ids = ["new_input"]
        self.index = {"new_input": 0}
        self.graph_pairs = {"new_input": {position: graph_hashes[position] for position in ("lower", "upper")}}
        self.graph_cache = {graph_hashes[position]: graphs[position] for position in ("lower", "upper")}
        self.numeric_values = torch.from_numpy(numeric.astype(np.float32))

    def validate_ids(self, ids):
        assert list(ids) == ["new_input"]

    def numeric(self, scope):
        assert scope == "fit_core"
        return self.numeric_values


class FrozenPredictor:
    def __init__(self, contract_path=CONTRACT):
        self.contract_path = Path(contract_path)
        self.contract = read(self.contract_path)
        assert self.contract["status"] == "FROZEN" and self.contract["model_version"] == VERSION
        for relative, digest in self.contract["bindings_sha256"].items():
            assert sha(RP / relative) == digest, f"Frozen input/rule binding changed: {relative}"
        self.state = read(S8 / "preprocessing/states/fit_core.json")
        assert self.state["feature_names"] == self.contract["required_feature_names"]
        assert self.state["fit_scope"] == "fit_core" and self.state["profile"] == "available_initial_stacking"
        self.base = read(ROOT / "base_selection_freeze.json")
        self.probability = read(ROOT / "calibration/probability.json")
        self.decision = read(ROOT / "calibration/decision.json")
        self.interval = read(ROOT / "calibration/interval.json")
        self.preprocess = load_preprocessing()
        sys.path.insert(0, str(S10))
        from graph_engine import predict_arrays, set_seed
        from models import build_model
        set_seed(20261015)
        self.predict_arrays, self.build_model = predict_arrays, build_model
        self.loaded = {}

    def _base_output(self, record_id=None):
        return {"record_id": record_id, "prediction_status": "invalid_input", "protocol": None,
                "anderson_gap_eV": None, "anderson_signed_gap_eV": None,
                "p_anderson_failure": None, "raw_p_anderson_failure": None,
                "probability_threshold": self.decision["policy"]["threshold"],
                "anderson_decision": "unavailable", "route": "abstain", "service": self.contract["service"],
                "predicted_residual_eV": None, "predicted_PBE_gap_eV": None,
                "prediction_interval_eV": None, "additional_properties": {},
                "warnings": [], "errors": [], "model_version": VERSION,
                "prediction_contract_sha256": sha(self.contract_path), "input_hash": None}

    @staticmethod
    def _graph(path):
        with np.load(path, allow_pickle=False) as archive:
            required = {"atomic_numbers", "edge_index", "edge_rbf", "edge_vectors_A"}
            if not required <= set(archive.files):
                raise ValueError("graph NPZ missing required arrays")
            z = np.asarray(archive["atomic_numbers"])
            edge = np.asarray(archive["edge_index"])
            rbf = np.asarray(archive["edge_rbf"])
            vectors = np.asarray(archive["edge_vectors_A"])
        if z.ndim != 1 or not len(z) or not np.isfinite(z).all() or not np.equal(z, np.round(z)).all():
            raise ValueError("invalid graph atomic_numbers")
        if np.any((z < 1) | (z > 118)):
            raise ValueError("graph atomic number outside fixed Z=1..118 vocabulary")
        if edge.ndim != 2 or edge.shape[0] != 2 or not edge.shape[1] or not np.isfinite(edge).all() or not np.equal(edge, np.round(edge)).all():
            raise ValueError("invalid graph edge_index")
        if np.any((edge < 0) | (edge >= len(z))):
            raise ValueError("graph edge_index outside node range")
        if rbf.shape != (edge.shape[1], 32) or vectors.shape != (edge.shape[1], 3):
            raise ValueError("graph edge arrays differ from frozen 32-RBF schema")
        if not np.isfinite(rbf).all() or not np.isfinite(vectors).all():
            raise ValueError("nonfinite graph edge arrays")
        return {"atomic_numbers": torch.from_numpy(z.astype(np.int64)),
                "edge_index": torch.from_numpy(edge.astype(np.int64)),
                "edge_rbf": torch.from_numpy(rbf.astype(np.float32)),
                "edge_vectors_A": torch.from_numpy(vectors.astype(np.float32))}

    def _heads(self, adapter):
        output = {}
        internal = {"classifier": "classifier", "residual_regression": "residual", "direct_gap_regression": "direct_gap"}
        for task, entries in self.contract["selected_models"].items():
            values = []
            for entry in entries:
                key = entry["path"]
                if key not in self.loaded:
                    path = S10 / key
                    assert sha(path) == entry["sha256"], "Selected checkpoint hash changed"
                    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
                    spec = checkpoint["specification"]
                    assert spec["fit_scope"] == "fit_core"
                    assert spec["preprocessor_sha256"] == sha(S8 / "preprocessing/states/fit_core.json")
                    assert spec["models_sha256"] == sha(S10 / "models.py")
                    model = self.build_model(spec["family"], spec["params"], n_numeric=1540, tasks=spec["tasks"])
                    model.load_state_dict(checkpoint["model_state_dict"])
                    model.eval()
                    self.loaded[key] = (model, tuple(spec["tasks"]))
                model, tasks = self.loaded[key]
                prediction = self.predict_arrays(model, adapter, ["new_input"], "fit_core", tasks)
                values.append(float(prediction[internal[task]][0]))
            output[task] = float(np.mean(values))
        return output

    def predict(self, request, request_base=ROOT):
        output = self._base_output(request.get("record_id") if isinstance(request, dict) else None)
        try:
            if not isinstance(request, dict):
                raise ValueError("request must be a JSON object")
            if set(request) - {"record_id", "protocol", "raw_features", "graphs", "graph_format"}:
                raise ValueError("unexpected request fields; targets are never accepted")
            protocol = request.get("protocol")
            if not isinstance(protocol, dict) or any(protocol.get(key) != value for key, value in PROTOCOL.items()):
                raise ValueError("protocol does not match frozen PBE/noSOC/initial-candidate contract")
            if set(protocol) != set(PROTOCOL) | {"reference_track"} or protocol.get("soc") is not False:
                raise ValueError("protocol requires exactly the documented fields and boolean soc=false")
            if protocol.get("reference_track") not in ("native_free", "matched_strain"):
                raise ValueError("reference_track must be native_free or matched_strain")
            output["protocol"] = protocol
            if request.get("graph_format") != GRAPH_FORMAT:
                raise ValueError("graph_format differs from frozen Step08 graph construction")
            values = request.get("raw_features")
            names = self.state["feature_names"]
            if not isinstance(values, dict) or set(values) != set(names):
                missing = sorted(set(names) - set(values or {}))
                extra = sorted(set(values or {}) - set(names))
                raise ValueError(f"all 770 raw feature names required; missing={missing[:12]}, extra={extra[:12]}")
            raw = np.array([[np.nan if values[name] is None else float(values[name]) for name in names]], dtype=float)
            if any(value is not None and not np.isfinite(float(value)) for value in values.values()):
                raise ValueError("nonnull raw feature values must be finite")
            if any(values[name] is None or not np.isfinite(float(values[name])) for name in self.contract["critical_finite_feature_names"]):
                raise ValueError("critical gap/contact-edge/contact-side/Anderson input is missing or nonfinite")
            if any(values[f"{position}__gap_eV"] <= 0 for position in ("lower", "upper")):
                raise ValueError("both isolated monolayer gaps must be strictly positive")
            for position in ("lower", "upper"):
                if values[f"{position}__contact_side_canonical_code"] not in (-1, 0, 1):
                    raise ValueError("contact-side canonical code must be -1, 0 (equivalent faces), or +1")
                if values[f"{position}__contact_side_canonical_code"] == 0 and values[f"{position}__face_equivalent"] != 1:
                    raise ValueError("contact code 0 requires equivalent-face input")
                if values[f"{position}__cbm_contact_eV"] <= values[f"{position}__vbm_contact_eV"]:
                    raise ValueError("isolated monolayer contact edges must have a positive gap")
                if abs(values[f"{position}__gap_eV"] - values[f"{position}__cbm_contact_eV"] + values[f"{position}__vbm_contact_eV"]) > 1e-6:
                    raise ValueError("isolated monolayer gap disagrees with its contact edges")
            signed = min(values["lower__cbm_contact_eV"], values["upper__cbm_contact_eV"]) - max(values["lower__vbm_contact_eV"], values["upper__vbm_contact_eV"])
            gA = max(0., signed)
            if abs(values["anderson_signed_eV"] - signed) > 1e-6 or abs(values["anderson_nonnegative_eV"] - gA) > 1e-6:
                raise ValueError("Anderson values disagree with contact edges by more than 1e-6 eV")
            track = values["reference_track_matched_strain"]
            if track != float(protocol["reference_track"] == "matched_strain"):
                raise ValueError("reference_track metadata and numeric input disagree")
            graph_entries = request.get("graphs")
            if not isinstance(graph_entries, dict) or set(graph_entries) != {"lower", "upper"}:
                raise ValueError("lower and upper graph NPZ inputs required")
            graphs, graph_hashes = {}, {}
            for position, entry in graph_entries.items():
                if not isinstance(entry, dict) or set(entry) - {"path", "sha256"} or "path" not in entry:
                    raise ValueError("graph entries require path and optional sha256")
                path = local_path(entry["path"], request_base)
                digest = sha(path)
                if entry.get("sha256", digest) != digest:
                    raise ValueError("input graph SHA mismatch")
                graph_hashes[position] = digest
                graphs[position] = self._graph(path)
                z = graphs[position]["atomic_numbers"].numpy()
                for atomic_number in range(1, 119):
                    fraction = values[f"{position}__atomic_fraction_Z{atomic_number}"]
                    if fraction is None or abs(fraction - float(np.mean(z == atomic_number))) > 1e-6:
                        raise ValueError("numeric layer composition disagrees with graph atomic numbers")
            input_hash_payload = {"request": request, "graph_sha256": graph_hashes}
            output["input_hash"] = hashlib.sha256(json.dumps(input_hash_payload, sort_keys=True, ensure_ascii=False,
                                                             separators=(",", ":"), allow_nan=False).encode()).hexdigest()
            columns = {column["name"]: column for column in self.state["columns"]}
            elements = sorted({int(z) for graph in graphs.values() for z in graph["atomic_numbers"]})
            unseen = [z for z in elements if all(columns[f"{position}__atomic_fraction_Z{z}"]["constant"] and
                                                columns[f"{position}__atomic_fraction_Z{z}"]["median"] == 0.
                                                for position in ("lower", "upper"))]
            transformed = self.preprocess.transform(raw, self.state)
            unsupported = [names[i] for i in np.flatnonzero(transformed["unsupported_feature_mask"][0])]
            if unseen or unsupported or values["twist_sin"] is not None or values["twist_cos"] is not None:
                output.update(prediction_status="out_of_domain", unseen_element_Z=unseen,
                              unsupported_feature_names=unsupported,
                              warnings=["Input includes elements or conditions without training support; numerical prediction withheld."])
                return output
            adapter = InputAdapter(transformed["graph_values"], graphs, graph_hashes)
            heads = self._heads(adapter)
            raw_p = np.clip(heads["classifier"], *self.probability["input_clip"])
            logit = np.log(raw_p / (1. - raw_p))
            calibrated_logit = self.probability["a"] * logit + self.probability["b"]
            p = float(np.exp(-np.logaddexp(0., -calibrated_logit)))
            policy = self.decision["policy"]
            applicable = policy["policy"] != "reject_all" and p <= policy["threshold"]
            residual_raw = gA + heads["residual_regression"]
            direct_raw = heads["direct_gap_regression"]
            residual_gap, direct_gap = max(0., residual_raw), max(0., direct_raw)
            comparator_gap = residual_gap if applicable else direct_gap
            service = self.contract["service"]
            gap = {"full_domain_residual": residual_gap, "full_domain_direct": direct_gap,
                   "classifier_routed": comparator_gap}[service]
            interval = self.interval["pipelines"][service]
            q = interval["q_eV"]
            bounds = [max(0., gap - q), gap + q] if q is not None else [0., None]
            warning = ["Input passes frozen support checks; this status is not a predictive reliability guarantee."]
            optional_missing = [name for name, value in values.items() if value is None]
            if optional_missing:
                warning.append("Optional null inputs use frozen fit_core imputation and missing indicators.")
            if protocol["reference_track"] == "matched_strain":
                warning.append("Prediction is conditional on matched-strain monolayer reference inputs.")
            output.update(prediction_status="in_domain", anderson_gap_eV=gA, anderson_signed_gap_eV=signed,
                          p_anderson_failure=p, raw_p_anderson_failure=heads["classifier"],
                          anderson_decision="predicted_applicable" if applicable else "not_accepted_as_applicable",
                          route=service, predicted_residual_eV=heads["residual_regression"],
                          predicted_PBE_gap_eV=gap, prediction_interval_eV=bounds,
                          interval_nominal_coverage=.9, interval_unit=self.interval["unit"],
                          interval_q_eV=q, interval_unbounded=interval["unbounded"],
                          raw_residual_branch_gap_eV=residual_raw, raw_direct_gap_eV=direct_raw,
                          residual_branch_gap_eV=residual_gap, direct_branch_gap_eV=direct_gap,
                          nonnegative_projection_applied={"residual": residual_raw < 0, "direct": direct_raw < 0},
                          routed_comparator={"route": "anderson_plus_residual" if applicable else "direct_regression",
                                             "predicted_PBE_gap_eV": comparator_gap},
                          optional_missing_feature_names=optional_missing, unseen_element_Z=[],
                          unsupported_feature_names=[], warnings=warning)
            return output
        except (ValueError, TypeError, KeyError, OSError, OverflowError) as error:
            output["errors"] = [str(error)]
            return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("request", nargs="?")
    parser.add_argument("--output")
    parser.add_argument("--freeze-contract", action="store_true")
    args = parser.parse_args()
    if args.freeze_contract:
        contract = freeze_contract()
        print(json.dumps({"status": contract["status"], "prediction_contract_sha256": sha(CONTRACT)}))
    else:
        if not args.request:
            parser.error("request JSON is required")
        path = local_path(args.request, Path.cwd())
        result = FrozenPredictor().predict(read(path), path.parent)
        if args.output:
            save(local_path(args.output, Path.cwd()), result)
        print(json.dumps(result, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
