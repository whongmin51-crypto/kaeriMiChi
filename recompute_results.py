"""Recompute all heldout service metrics/intervals/CIs from retained predictions."""
from pathlib import Path
import os
for name in ['OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS']:
    os.environ[name]='1'
import argparse
import csv
import hashlib
import json
import math
import sys
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'code'))
from common12 import read, rows, rm, cm
from evaluate_arrays import service_analysis, curves


def compare(actual, expected, location='root'):
    if isinstance(expected,dict):
        assert set(actual)==set(expected),(location,'keys')
        return sum(compare(actual[k],expected[k],location+'.'+k) for k in expected)
    if isinstance(expected,list):
        assert len(actual)==len(expected),(location,'length')
        return sum(compare(a,b,f'{location}[{i}]') for i,(a,b) in enumerate(zip(actual,expected)))
    if isinstance(expected,(float,int)) and not isinstance(expected,bool):
        assert actual is not None and math.isclose(float(actual),float(expected),rel_tol=1e-11,abs_tol=1e-12),(location,actual,expected)
    else:
        assert actual==expected,(location,actual,expected)
    return 1


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=ROOT/'recomputed');a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=True)
    verified_files=0
    for item in read(ROOT/'copy_manifest.json'):
        actual=hashlib.sha256((ROOT/item['path']).read_bytes()).hexdigest()
        assert actual==item['sha256'],item['path'];verified_files+=1
    checks=[];table=[]
    for role in ['sealed_final_test','external_validation_diagnostic']:
        evaluated=rows(ROOT/'data/evaluation'/role/'evaluated_predictions.jsonl')
        rr=[r['metadata'] for r in evaluated];primary=[r['frozen_prediction'] for r in evaluated]
        assert len(rr)==len(set(r['heterostructure_id'] for r in rr))
        assert set(r.get('source',r.get('dataset')) for r in rr)<=set(['Janus','HetDB'])
        assert [r['heterostructure_id'] for r in rr]==read(ROOT/'data/role_ids.json')[role]
        actual=service_analysis(rr,primary)
        expected=read(ROOT/'data/evaluation'/role/'main_metrics.json')
        values=compare(actual,expected)
        (a.output/(role+'_main_metrics.json')).write_text(json.dumps(actual,indent=2,allow_nan=False),encoding='utf8')
        values+=compare(curves(rr,primary),rows(ROOT/'data/evaluation'/role/'risk_coverage.jsonl'))
        raw=[]
        for file in sorted((ROOT/'data/inference'/role).glob('*_ensemble.jsonl')):
            batch=rows(file)
            assert [r['heterostructure_id'] for r in batch]==[r['heterostructure_id'] for r in rr]
            candidate=file.name.removesuffix('_ensemble.jsonl');item={'candidate':candidate}
            if candidate.endswith(('multi_task','classifier')):item['classification']=cm(rr,[r['p_fail'] for r in batch])
            if candidate.endswith('multi_task'):
                item['residual']=rm(rr,[r['predicted_residual_gap_eV'] for r in batch])
                item['direct']=rm(rr,[r['predicted_direct_gap_eV'] for r in batch])
            elif not candidate.endswith('classifier'):
                item['regression']=rm(rr,[r['predicted_gap_eV'] for r in batch])
            raw.append(item)
        values+=compare(raw,read(ROOT/'data/evaluation'/role/'four_model_comparison.json')['candidates'])
        for name in ['service','routed','direct','Anderson_on_service_cohort']:
            met=actual['regression'][name]['metrics']
            for source in ['source_macro']+sorted(met['by_source']):
                metric=met['source_macro'] if source=='source_macro' else met['by_source'][source]
                table.append({'role':role,'pipeline':name,'source':source,'MAE_eV':metric['mae_eV'],'RMSE_eV':metric['rmse_eV']})
        checks.append({'role':role,'rows':len(rr),'matched_scalar_values':values,'status':'PASS'})
    with (a.output/'gap_errors.csv').open('w',encoding='utf8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(table[0]));writer.writeheader();writer.writerows(table)
    result={'status':'PASS','scope':'result recomputation from archived fixed predictions, not training or checkpoint inference',
            'source_files_verified':verified_files,'checks':checks,
            'bootstrap_replicates':1000,'bootstrap_seed':20261025,'absolute_tolerance':1e-12,'relative_tolerance':1e-11}
    (a.output/'validation.json').write_text(json.dumps(result,indent=2),encoding='utf8')
    print(json.dumps(result))


if __name__=='__main__':main()
