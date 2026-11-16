from common import *
from collections import Counter
import numpy as np

def main():
    assert not (ROOT/'protocol.json').exists(),'Run registration is immutable; do not overwrite.'
    original=rows(S7/'record_partitions.jsonl')
    kept=[r for r in original if r['dataset']!='Tony' and r['partition'] in ROLES]
    assert all(r['eligible_for_assigned_role'] for r in kept)
    pools={role:[r['heterostructure_id'] for r in kept if r['partition']==role] for role in ROLES}
    groups={role:{r['split_group_key'] for r in kept if r['partition']==role} for role in ROLES}
    for i,a in enumerate(ROLES):
        for b in ROLES[i+1:]: assert not groups[a]&groups[b],(a,b)
    core=[r for r in read(S9/'data_cache/core_metadata.json') if r['dataset']!='Tony']
    fitids=[r['heterostructure_id'] for r in core]
    assert set(fitids)==set(pools['fit_core']) and len(fitids)==11561
    write_rows(ROOT/'inputs/partitions.jsonl',kept);save(ROOT/'inputs/core_metadata.json',core)
    save(ROOT/'inputs/role_ids.json',pools)
    allids=[r['heterostructure_id'] for r in kept]
    with np.load(S8/'features/raw_feature_matrix.npz',allow_pickle=False) as ar:
        ix={str(k):i for i,k in enumerate(ar['record_ids'])};names=ar['feature_names'].astype(str).tolist()
        raw=ar['values'][[ix[k] for k in allids]].copy()
    np.savez_compressed(ROOT/'inputs/raw_features.npz',record_ids=np.array(allids),feature_names=np.array(names),values=raw)
    pp=module('no_tony_preprocessor',S8/'preprocessing/preprocessing.py');ix={k:i for i,k in enumerate(allids)}
    state=pp.fit_preprocessor(raw[[ix[k] for k in fitids]],names,fitids)
    state['profile']='available_initial_stacking';save(ROOT/'inputs/fit_core_preprocessor.json',state)
    selected=set(allids)
    gp=[r for r in rows(S8/'features/features.jsonl') if r['heterostructure_id'] in selected]
    save(ROOT/'inputs/graph_pairs.json',{r['heterostructure_id']:r['graph_ids'] for r in gp})
    needed={g for r in gp for g in r['graph_ids'].values()}
    gm=[r for r in rows(S8/'graphs/cache_manifest.jsonl') if r['graph_id'] in needed]
    save(ROOT/'inputs/graph_manifest.json',gm)
    jobs=[];protected={}
    for stage,backend in [(S10,'graph'),(S9,'trees')]:
        for e in read(stage/'final_models_manifest.json')['models']:
            old=stage/e['path'];assert sha(old)==e['sha256']
            sp=old.parent/('fit_specification.json' if backend=='graph' else 'job.json');s=read(sp)
            j={'candidate':e['model']+'_'+e['task'],'family':e['model'],'task':e['task'],'seed':e['seed'],
               'backend':backend,'params':s['params'],'iterations':s['epochs_cap'] if backend=='graph' else e['selected_iterations'],
               'original_checkpoint':old.relative_to(RP).as_posix(),'original_sha256':e['sha256'],
               'original_spec':sp.relative_to(RP).as_posix(),'original_spec_sha256':sha(sp)}
            if backend=='graph':j.update(internal_task=s['task'],loss_weights=s['loss_weights'],patience=s['patience'])
            jobs.append(j);protected[j['original_checkpoint']]=e['sha256'];protected[j['original_spec']]=sha(sp)
    adopted=['GANN_residual_regression','GANN_multi_task','CGCNN_multi_task']
    jobs.sort(key=lambda j:(j['candidate'] not in adopted,adopted.index(j['candidate']) if j['candidate'] in adopted else 99,j['candidate'],j['seed']))
    for p in [S7/'record_partitions.jsonl',S7/'cv_folds.json',S8/'features/raw_feature_matrix.npz',S8/'preprocessing/preprocessing.py',S8/'graphs/cache_manifest.jsonl',S9/'engine.py',S9/'metrics.py',S9/'final_models_manifest.json',S10/'graph_engine.py',S10/'graph_data.py',S10/'models.py',S10/'final_models_manifest.json',S11/'base_selection_freeze.json',S11/'calibrate11.py',S11/'common11.py']:
        protected[p.relative_to(RP).as_posix()]=sha(p)
    save(ROOT/'upstream_hashes.json',protected);save(ROOT/'jobs.json',jobs)
    protocol={'status':'REGISTERED_BEFORE_TRAINING','created_utc':now(),'user_instruction':'Drop Tony; keep current hyperparameters, retrain from scratch, recalibrate and evaluate, then remove old results and duplicated descriptions from current paper.',
      'sources':['Janus','HetDB'],'excluded_source':'Tony','roles':{k:{'n':len(v),'ids_sha256':idsha(v),'composition_groups':len(groups[k])} for k,v in pools.items()},
      'source_counts':{r:dict(Counter(x['dataset'] for x in kept if x['partition']==r)) for r in ROLES},
      'model_count':70,'graph_models':40,'tree_models':30,'random_seeds':list(range(20261015,20261020)),
      'hyperparameter_policy':'Exact original final params, fixed selected epochs/iterations, seeds and multitask loss weights; no new search, early-stop selection or head selection.',
      'data_policy':'Remove Tony from every role; remaining per-ID roles unchanged; no resampling or role transfer. Refit preprocessing and fit-dependent loss scales on remaining fit_core only.',
      'calibration_policy':'Same global monotone Platt, 1e-4 penalty, source/pair weighting, 5% empirical decision target/minimum 10 composition groups, group-max 90% interval procedure; re-estimate on disjoint filtered calibration identities.',
      'selected_heads':{k:{'candidate':v['candidate'],'head_field':v['head_field']} for k,v in read(S11/'base_selection_freeze.json')['heads'].items()},
      'service':'full_domain_residual','evaluation_policy':'All new predictions and calibration frozen before opening heldout targets. Same existing Janus/HetDB heldouts are repeated evaluation, not a newly unseen prospective test. No Tony evaluation or new model selection.',
      'manuscript_policy':'Only after new training/calibration/evaluation acceptance: replace active paper old ML results and duplicate descriptions. Preserve original scientific records and raw data.',
      'input_feature_count':770,'model_feature_count':1540,'input_profile':'available_initial_stacking','fit_core_id_sha256':idsha(fitids),
      'jobs_sha256':sha(ROOT/'jobs.json'),'upstream_hashes_sha256':sha(ROOT/'upstream_hashes.json'),
      'inputs_sha256':{p.relative_to(ROOT).as_posix():sha(p) for p in (ROOT/'inputs').iterdir() if p.is_file()},
      'historical_configuration_origin':'The retained hyperparameters were originally selected in the three-source study; this is not a claim of development independent of Tony.'}
    save(ROOT/'protocol.json',protocol)
    save(ROOT/'validation.json',{'status':'REGISTERED_NOT_COMPLETE','training':False,'calibration':False,'evaluation':False,'manuscript_replaced':False})
    print(json.dumps({'counts':protocol['source_counts'],'jobs':len(jobs),'graphs':len(gm)},ensure_ascii=False))
if __name__=='__main__':main()
