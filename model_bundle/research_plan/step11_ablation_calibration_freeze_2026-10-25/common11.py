"""New Step11: explicit role access, immutable upstream bindings, array metrics."""
from pathlib import Path
from datetime import datetime, timezone
import json, hashlib, sys, re
import numpy as np

ROOT = Path(__file__).resolve().parent
RP = ROOT.parent
S7 = RP/'step07_2026-10-15'
S8 = RP/'step08_2026-10-15'
S9 = RP/'step09_2026-10-15'
S10 = RP/'step10_2026-10-15'
sys.path.insert(0, str(S9))
from metrics import classification_metrics, regression_metrics, source_pair_weights

def read(p): return json.loads(Path(p).read_text(encoding='utf-8'))
def rows(p):
    with Path(p).open(encoding='utf-8') as f: return [json.loads(s) for s in f if s.strip()]
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024), b''): h.update(b)
    return h.hexdigest()
def save(p,v):
    p=Path(p); p.parent.mkdir(parents=True,exist_ok=True)
    q=p.with_name(p.name+'.tmp')
    q.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8');q.replace(p)
def write_rows(p,rr):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(''.join(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n' for r in rr),encoding='utf-8')
def now(): return datetime.now(timezone.utc).isoformat()
def idsha(ids): return hashlib.sha256(json.dumps(sorted(ids),ensure_ascii=False,separators=(',',':')).encode()).hexdigest()
def partitions(): return rows(S7/'record_partitions.jsonl')
def role_ids(role):
    assert role in ('fit_core','cal_probability','cal_decision','cal_interval')
    return [r['heterostructure_id'] for r in partitions() if r['partition']==role]
def load_role_labels(role):
    """Only one authorized train-role subset, with an append-only access receipt."""
    ids=role_ids(role);selected=set(ids);out={}
    with (RP/'step06_2026-10-15/labels/train_labels.jsonl').open(encoding='utf-8') as f:
        for line in f:
            # Select the identity token before parsing any target in this line.
            token=re.search(r'"heterostructure_id"\s*:\s*"([^"\\]+)"',line)
            if token and token.group(1) in selected:
                r=json.loads(line)
                assert r['masks']['main_operational'] and r['split_role']=='train'
                out[r['heterostructure_id']]=r
    assert set(out)==selected
    with (ROOT/'label_access_log.jsonl').open('a',encoding='utf-8') as f:
        f.write(json.dumps({'at_utc':now(),'role':role,'ids_sha256':idsha(ids),'count':len(ids),
            'file':'step06_2026-10-15/labels/train_labels.jsonl','external_or_test_opened':False})+'\n')
    return [out[i] for i in ids]
def weights(rr, source_equal=True):
    return source_pair_weights([r['material_pair_id'] for r in rr], [r.get('source',r.get('dataset')) for r in rr] if source_equal else None)
def cm(rr,p): return classification_metrics([r['y'] for r in rr],p,[r['material_pair_id'] for r in rr],[r.get('source',r.get('dataset')) for r in rr])
def rm(rr,p): return regression_metrics([r['g_PBE_eV'] for r in rr],p,[r['material_pair_id'] for r in rr],[r.get('source',r.get('dataset')) for r in rr])
def curve(rr,p):
    p=np.asarray(p);w=weights(rr);y=np.array([r['y'] for r in rr]);out=[]
    for t in np.unique(p):
        a=p<=t;mass=float(w[a].sum())
        out.append({'threshold':float(t),'weighted_coverage':mass,'empirical_UAR':float(np.dot(w[a],y[a])/mass),
            'accepted_records':int(a.sum()),'accepted_failure_records':int(y[a].sum()),
            'accepted_composition_groups':len({r['split_group_key'] for r,b in zip(rr,a) if b})})
    return out
def choose_threshold(cc):
    ok=[r for r in cc if r['empirical_UAR']<=.05+1e-12 and r['accepted_composition_groups']>=10]
    if not ok:return {'policy':'reject_all','threshold':None,'weighted_coverage':0.,'empirical_UAR':None,'accepted_records':0,'accepted_composition_groups':0}
    r=min(ok,key=lambda r:(-r['weighted_coverage'],r['empirical_UAR'],r['threshold']))
    return {'policy':'threshold',**r}
def accepted(p,policy):
    return np.zeros(len(p),dtype=bool) if policy['policy']=='reject_all' else np.asarray(p)<=policy['threshold']
