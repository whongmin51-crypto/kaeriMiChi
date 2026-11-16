"""Portable array-metric I/O; no upstream project paths or fitting functions."""
from pathlib import Path
import json
import numpy as np
from metrics import classification_metrics, regression_metrics, source_pair_weights

ROOT = Path(__file__).resolve().parents[1] / 'data'
S11 = ROOT

def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))

def rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf8').splitlines() if line.strip()]

def weights(rr):
    return source_pair_weights([r['material_pair_id'] for r in rr], [r.get('source',r.get('dataset')) for r in rr])

def cm(rr,p):
    return classification_metrics([r['y'] for r in rr],p,[r['material_pair_id'] for r in rr],[r.get('source',r.get('dataset')) for r in rr])

def rm(rr,p):
    return regression_metrics([r['g_PBE_eV'] for r in rr],p,[r['material_pair_id'] for r in rr],[r.get('source',r.get('dataset')) for r in rr])
