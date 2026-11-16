"""Replay archived calibration/holdout inputs with the exact current checkpoints."""
from pathlib import Path
import os
for name in ['OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS']:
    os.environ[name]='1'
import argparse
import gc
import importlib.metadata
import json
import sys
import time
import numpy as np
ROOT=Path(__file__).resolve().parent
BUNDLE=ROOT/'model_bundle'
RP=BUNDLE/'research_plan'
RUN=RP/'retrain_no_tony_2026-11-02'
sys.path.insert(0,str(RUN))
from common import read, rows, sha, module
from run_models import Inputs


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--backend',choices=['trees','graph'],required=True)
    parser.add_argument('--output',type=Path,default=ROOT/'replayed')
    a=parser.parse_args();a.output.mkdir(exist_ok=True,parents=True)
    start=time.perf_counter();data=Inputs(graph=a.backend=='graph')
    jobs=[j for j in read(RUN/'jobs.json') if j['backend']==a.backend]
    ids=[i for role in ['cal_probability','cal_decision','cal_interval','sealed_final_test','external_validation_diagnostic'] for i in data.roles[role]]
    positions={key:i for i,key in enumerate(ids)}
    accum={};checks=[]
    if a.backend=='graph':
        sys.path.insert(0,str(RP/'step10_2026-10-15'))
        from graph_engine import predict_model,set_seed
        set_seed(20261015)
    else:
        engine=module('portable_tree_engine',RP/'step09_2026-10-15/engine.py')
        X=data.tree_values[[data.index[i] for i in ids]]
    for job in jobs:
        folder=RUN/'models'/job['candidate']/('seed'+str(job['seed']))
        receipt=read(folder/'receipt.json');checkpoint=folder/receipt['checkpoint']
        assert sha(checkpoint)==receipt['checkpoint_sha256']
        if a.backend=='graph':
            pred=predict_model(data,folder,ids);pred.pop('ids')
        else:
            fitted=engine.load_model(checkpoint,job['family'])
            values=engine.predict_head(fitted,job['family'],job['task'],X)
            key={'classifier':'classifier','residual_regression':'residual','direct_gap_regression':'direct_gap'}[job['task']]
            pred={key:values};del fitted
        accum.setdefault(job['candidate'],[]).append(pred)
        checks.append({'candidate':job['candidate'],'seed':job['seed'],'checkpoint_sha256':sha(checkpoint),'rows':len(ids)})
        print('REPLAY',job['candidate'],job['seed'],flush=True)
        gc.collect()
    ensemble_checks=[]
    for candidate,batch in sorted(accum.items()):
        assert len(batch)==5
        average={key:np.mean([p[key] for p in batch],axis=0) for key in batch[0]}
        maximum=0.;count=0
        for role in ['cal_probability','cal_decision','cal_interval','sealed_final_test','external_validation_diagnostic']:
            expected=rows(ROOT/'data/inference'/role/(candidate+'_ensemble.jsonl'))
            for row in expected:
                i=positions[row['heterostructure_id']]
                mapping={'classifier':'p_fail','residual':'predicted_residual_eV','direct_gap':'predicted_direct_raw_eV'} if candidate.endswith('multi_task') else {next(iter(average)):'raw_head'}
                for key,field in mapping.items():
                    delta=abs(float(average[key][i])-row[field]);maximum=max(maximum,delta);count+=1
                    assert np.isclose(average[key][i],row[field],rtol=1e-7,atol=1e-7),(candidate,role,row['heterostructure_id'],key,delta)
        ensemble_checks.append({'candidate':candidate,'head_values_checked':count,'max_absolute_difference':maximum,'status':'PASS'})
    packages=['numpy','torch'] if a.backend=='graph' else ['numpy','scikit-learn','xgboost','joblib']
    result={'status':'PASS','backend':a.backend,'models_replayed':len(checks),'prediction_rows_per_model':len(ids),
            'scope':'all five calibration/holdout roles, fixed raw heads; no fitting, tuning or recalibration',
            'ensembles':ensemble_checks,'checkpoints':checks,'elapsed_seconds':time.perf_counter()-start,
            'python':sys.version,'packages':{p:importlib.metadata.version(p) for p in packages},
            'absolute_tolerance':1e-7,'relative_tolerance':1e-7}
    (a.output/(a.backend+'_validation.json')).write_text(json.dumps(result,indent=2),encoding='utf8')
    print(json.dumps({k:result[k] for k in ['status','backend','models_replayed','prediction_rows_per_model','elapsed_seconds']}),flush=True)


if __name__=='__main__':main()
