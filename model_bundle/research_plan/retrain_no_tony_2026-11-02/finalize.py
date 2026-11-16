"""After all fixed fits: ensemble, recalibrate, seal, evaluate and audit.

There is no training or model-selection operation in this program.
"""
from common import *
import re, argparse
import numpy as np
from run_models import Inputs

def setup_calibration():
    # Import unchanged mathematical procedures with explicitly redirected I/O.
    sys.path.insert(0,str(S11))
    import common11
    common11.ROOT=ROOT
    common11.partitions=lambda: rows(ROOT/'inputs/partitions.jsonl')
    common11.save=save;common11.write_rows=write_rows
    import calibrate11
    assert calibrate11.ROOT==ROOT
    return common11,calibrate11

def build_ensembles(data):
    jobs=read(ROOT/'jobs.json');candidates=sorted({j['candidate'] for j in jobs})
    files={};receipts=[]
    for candidate in candidates:
        subset=[j for j in jobs if j['candidate']==candidate]
        assert len(subset)==5 and sorted(j['seed'] for j in subset)==list(range(20261015,20261020))
        preds=[];ids=None
        for j in subset:
            verify_job(j);out=jobdir(j);r=read(out/'receipt.json');assert r['status']=='PASS'
            assert sha(out/r['checkpoint'])==r['checkpoint_sha256'] and sha(out/'predictions.npz')==r['predictions_sha256']
            assert r['protocol_sha256']==sha(ROOT/'protocol.json')
            with np.load(out/'predictions.npz',allow_pickle=False) as ar:
                thisids=ar['ids'].astype(str).tolist();assert ids is None or thisids==ids;ids=thisids
                preds.append({k:ar[k].copy() for k in ar.files if k!='ids'})
            receipts.append(r)
        ensemble={k:np.mean([p[k] for p in preds],axis=0) for k in preds[0]};index={k:i for i,k in enumerate(ids)}
        for role in ROLES[1:]:
            rr=[]
            for key in data.roles[role]:
                i=index[key];r=data.partition[key];a=float(data.g_A[data.index[key]])
                z={'heterostructure_id':key,'source':r['dataset'],'material_pair_id':r['material_pair_id'],'split_group_key':r['split_group_key'],'g_A_eV':a,'ensemble_seed_count':5}
                if candidate.endswith('multi_task'):
                    z.update(p_fail=float(ensemble['classifier'][i]),predicted_residual_eV=float(ensemble['residual'][i]),predicted_direct_raw_eV=float(ensemble['direct_gap'][i]))
                    z.update(predicted_residual_gap_eV=max(0,a+z['predicted_residual_eV']),predicted_direct_gap_eV=max(0,z['predicted_direct_raw_eV']))
                else:
                    k=next(iter(ensemble));z['raw_head']=float(ensemble[k][i])
                    if k=='classifier':z['p_fail']=z['raw_head']
                    else:z['predicted_gap_eV']=max(0,z['raw_head']+(a if k=='residual' else 0))
                rr.append(z)
            p=ROOT/'inference'/role/(candidate+'_ensemble.jsonl');write_rows(p,rr);files[p.relative_to(ROOT).as_posix()]=sha(p)
    save(ROOT/'inference_manifest.json',{'status':'PASS','created_utc':now(),'model_count':70,'ensemble_count':14,'files_sha256':files,'heldout_labels_read':False})
    save(ROOT/'fit_acceptance.json',{'status':'PASS','actual_new_fits':len(receipts),'completed_utc':now(),'same_hyperparameters_all':all(r['same_hyperparameters'] for r in receipts),'total_fit_seconds':sum(r['fit_seconds'] for r in receipts),'receipts':receipts})

def support(data,key):
    i=data.index[key];raw=data.raw[i];col={r['name']:r for r in data.state_value['columns']};ix={k:n for n,k in enumerate(data.names)}
    unseen=[z for z in range(1,119) if any(raw[ix[f'{pos}__atomic_fraction_Z{z}']]>0 for pos in ('lower','upper')) and all(col[f'{pos}__atomic_fraction_Z{z}']['constant'] and col[f'{pos}__atomic_fraction_Z{z}']['median']==0 for pos in ('lower','upper'))]
    unsupported=[n for n,v in zip(data.names,raw) if np.isfinite(v) and col[n]['all_missing']]
    twist=any(np.isfinite(raw[ix[n]]) for n in ('twist_sin','twist_cos'))
    return {'prediction_status':'out_of_domain' if unseen or unsupported or twist else 'in_domain','unseen_element_Z':unseen,'unsupported_feature_names':unsupported,'true_twist':bool(twist)}

def primary_and_seal(data,cal):
    prob=read(ROOT/'calibration/probability.json');dec=read(ROOT/'calibration/decision.json');ints=read(ROOT/'calibration/interval.json')
    for role in ROLES[-2:]:
        pred=cal.selected_predictions(role) if False else None
        heads={k:{r['heterostructure_id']:r for r in rows(ROOT/'inference'/role/(v['candidate']+'_ensemble.jsonl'))} for k,v in read(ROOT/'protocol.json')['selected_heads'].items()}
        out=[]
        for key in data.roles[role]:
            r=support(data,key);r['heterostructure_id']=key
            if r['prediction_status']=='in_domain':
                p0=heads['classifier'][key]['p_fail'];p=float(cal.apply_probability([p0],prob)[0]);gA=float(data.g_A[data.index[key]])
                rg=max(0,gA+heads['residual_regression'][key]['raw_head']);dg=max(0,heads['direct_gap_regression'][key]['predicted_direct_raw_eV']);accept=bool(cal.accepted([p],dec['policy'])[0]);route=rg if accept else dg
                r.update(raw_p_anderson_failure=p0,p_anderson_failure=p,anderson_gap_eV=gA,predicted_PBE_gap_eV=rg,direct_branch_gap_eV=dg,routed_comparator={'predicted_PBE_gap_eV':route},accepted=accept)
            out.append(r)
        write_rows(ROOT/'primary_predictions'/(role+'.jsonl'),out)
    files={}
    for folder in ('models','inference','primary_predictions','calibration','inputs'):
        for p in (ROOT/folder).rglob('*'):
            if p.is_file():files[p.relative_to(ROOT).as_posix()]=sha(p)
    for p in [ROOT/'jobs.json',ROOT/'base_selection_freeze.json',ROOT/'inference_manifest.json',ROOT/'fit_acceptance.json']:
        files[p.relative_to(ROOT).as_posix()]=sha(p)
    save(ROOT/'prediction_seal.json',{'status':'SEALED','sealed_utc':now(),'protocol_sha256':sha(ROOT/'protocol.json'),'heldout_labels_opened_in_this_run':False,'files_sha256':files})

def reveal(role):
    seal=read(ROOT/'prediction_seal.json');assert seal['status']=='SEALED'
    for p,h in seal['files_sha256'].items():assert sha(ROOT/p)==h,p
    assert sha(ROOT/'protocol.json')==seal['protocol_sha256']
    ids=read(ROOT/'inputs/role_ids.json')[role];chosen=set(ids);found={}
    src=RP/'step06_2026-10-15'/('sealed/test_labels.jsonl' if role=='sealed_final_test' else 'labels/validation_labels.jsonl')
    with src.open(encoding='utf-8') as f:
        for line in f:
            m=re.search(r'"heterostructure_id"\s*:\s*"([^"\\]+)"',line)
            if m and m.group(1) in chosen:
                r=json.loads(line);assert r['dataset']!='Tony' and r['masks']['main_operational'];found[m.group(1)]=r
    assert set(found)==chosen
    result=[found[k] for k in ids];write_rows(ROOT/'labels'/(role+'.jsonl'),result)
    with (ROOT/'label_access_log.jsonl').open('a',encoding='utf-8') as f:
        f.write(json.dumps({'role':role,'at_utc':now(),'rows':len(result),'prediction_seal_sha256':sha(ROOT/'prediction_seal.json'),'source_sha256':sha(src),'used_for_fitting_or_selection':False})+'\n')
    return result

def evaluation(cal):
    s12=RP/'step12_frozen_source_tests_2026-10-25';sys.path.insert(0,str(s12))
    import statistics12 as st
    import evaluate12 as ev
    ev.S11=ROOT
    # Historical routine assumes a non-null threshold; reject-all is represented
    # only in the array evaluation by -infinity, with the frozen policy untouched.
    dec=read(ROOT/'calibration/decision.json');real_read=ev.read
    def read_adapter(p):
        v=real_read(p)
        if Path(p)==ROOT/'calibration/decision.json' and v['policy']['policy']=='reject_all':v['policy']['threshold']=-float('inf')
        return v
    ev.read=read_adapter
    for role in ROLES[-2:]:
        rr=reveal(role);primary=rows(ROOT/'primary_predictions'/(role+'.jsonl'));assert [r['heterostructure_id'] for r in rr]==[r['heterostructure_id'] for r in primary]
        for r in rr:r['source']=r['dataset']
        root=ROOT/'evaluation'/role
        result=ev.service_analysis(rr,primary);save(root/'main_metrics.json',result)
        write_rows(root/'evaluated_predictions.jsonl',[{'metadata':r,'frozen_prediction':p} for r,p in zip(rr,primary)])
        write_rows(root/'risk_coverage.jsonl',ev.curves(rr,primary))
        strata=[]
        for axis in ('source','reference_track','y'):
            for v in sorted(set(r[axis] for r in rr)):
                ix=[i for i,r in enumerate(rr) if r[axis]==v]
                strata.append({'axis':axis,'level':v,'result':ev.service_analysis([rr[i] for i in ix],[primary[i] for i in ix])})
        save(root/'stratified_metrics.json',strata)
        bench=[]
        for p in sorted((ROOT/'inference'/role).glob('*_ensemble.jsonl')):
            batch=rows(p);assert [r['heterostructure_id'] for r in batch]==[r['heterostructure_id'] for r in rr]
            name=p.name.removesuffix('_ensemble.jsonl');rec={'candidate':name}
            if name.endswith('multi_task') or name.endswith('classifier'):rec['classification']=st.cm(rr,[r['p_fail'] for r in batch])
            if name.endswith('multi_task'):
                rec['residual']=st.rm(rr,[r['predicted_residual_gap_eV'] for r in batch]);rec['direct']=st.rm(rr,[r['predicted_direct_gap_eV'] for r in batch])
            elif not name.endswith('classifier'):rec['regression']=st.rm(rr,[r['predicted_gap_eV'] for r in batch])
            bench.append(rec)
        save(root/'four_model_comparison.json',{'scope':'raw ensemble predictions on all admitted rows; domain-screened service reported separately','model_selection_performed':False,'candidates':bench})

def acceptance():
    protected=read(ROOT/'upstream_hashes.json');changes=[p for p,h in protected.items() if sha(RP/p)!=h];assert not changes,changes
    jobs=read(ROOT/'jobs.json');assert len(jobs)==70
    state=read(ROOT/'inputs/fit_core_preprocessor.json');parts=rows(ROOT/'inputs/partitions.jsonl');pm={r['heterostructure_id']:r for r in parts}
    assert len(state['fit_ids'])==11561 and all(pm[k]['partition']=='fit_core' and pm[k]['dataset']!='Tony' for k in state['fit_ids'])
    assert all(r['dataset'] in ('Janus','HetDB') for r in parts)
    for j in jobs:
        verify_job(j);out=jobdir(j);r=read(out/'receipt.json');assert r['status']=='PASS' and r['fit_id_sha256']==idsha(state['fit_ids'])
        assert sha(out/r['checkpoint'])==r['checkpoint_sha256'] and r['checkpoint_sha256']!=j['original_sha256']
    for role in ROLES[-2:]:
        result=read(ROOT/'evaluation'/role/'main_metrics.json');assert result['rows']==len(read(ROOT/'inputs/role_ids.json')[role])
    save(ROOT/'validation.json',{'status':'SCIENTIFIC_PASS_MANUSCRIPT_PENDING','completed_utc':now(),'training':True,'actual_new_fits':70,'calibration':True,'evaluation':True,'manuscript_replaced':False,'upstream_files_verified_unchanged':len(protected),'protocol_sha256':sha(ROOT/'protocol.json'),'prediction_seal_sha256':sha(ROOT/'prediction_seal.json')})

def main():
    data=Inputs();build_ensembles(data)
    protocol=read(ROOT/'protocol.json');save(ROOT/'experiment_protocol.json',protocol)
    save(ROOT/'base_selection_freeze.json',{'status':'FROZEN_BEFORE_CALIBRATION','created_utc':protocol['created_utc'],'heads':protocol['selected_heads'],'service':protocol['service'],'protocol_sha256':sha(ROOT/'experiment_protocol.json'),'selection':'Inherited before this run; no re-selection'})
    common,cal=setup_calibration();cal.main();primary_and_seal(data,cal);evaluation(cal);acceptance()
    print('Scientific training/calibration/evaluation PASS; manuscript replacement remains pending.',flush=True)
if __name__=='__main__':main()
