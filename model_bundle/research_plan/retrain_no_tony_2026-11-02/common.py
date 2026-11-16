"""No-Tony fixed-configuration retraining. All writes stay in this run."""
from pathlib import Path
from datetime import datetime, timezone
import json, hashlib, os, sys, importlib.util
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
RP = ROOT.parent
S7 = RP/'step07_2026-10-15'
S8 = RP/'step08_2026-10-15'
S9 = RP/'step09_2026-10-15'
S10 = RP/'step10_2026-10-15'
S11 = RP/'step11_ablation_calibration_freeze_2026-10-25'
ROLES = ['fit_core','cal_probability','cal_decision','cal_interval','sealed_final_test','external_validation_diagnostic']
def now(): return datetime.now(timezone.utc).isoformat()
def read(p): return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def rows(p): return [json.loads(l) for l in Path(p).read_text(encoding='utf-8-sig').splitlines() if l.strip()]
def sha(p):
    with Path(p).open('rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()
def idsha(ids): return hashlib.sha256(json.dumps(sorted(ids),ensure_ascii=False,separators=(',',':')).encode()).hexdigest()
def save(p, v):
    p=Path(p).resolve(); assert p.is_relative_to(ROOT),p
    p.parent.mkdir(parents=True,exist_ok=True)
    t=p.with_name(p.name+'.'+str(os.getpid())+'.tmp')
    t.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8');t.replace(p)
def write_rows(p, rr):
    p=Path(p).resolve();assert p.is_relative_to(ROOT)
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(''.join(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n' for r in rr),encoding='utf-8')
def module(name,p):
    spec=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(spec)
    sys.modules[name]=m;spec.loader.exec_module(m);return m
def jobdir(j): return ROOT/'models'/j['candidate']/('seed'+str(j['seed']))
def verify_job(j):
    assert sha(RP/j['original_spec'])==j['original_spec_sha256']
    s=read(RP/j['original_spec']);assert s['params']==j['params'] and s['seed']==j['seed']
    if j['backend']=='graph':
        assert s['epochs_cap']==j['iterations'] and s['loss_weights']==j['loss_weights'] and s['patience']==j['patience']
    return s
