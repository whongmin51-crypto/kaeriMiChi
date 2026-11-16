"""Sequential probability, risk-constrained decision and group-interval freeze."""
from common11 import *
from scipy.optimize import minimize
from scipy.special import expit, logit
import csv, argparse

def selected_predictions(role):
    b=read(ROOT/'base_selection_freeze.json');ids=role_ids(role);out={}
    for task,s in b['heads'].items():
        path=ROOT/'inference'/role/(s['candidate']+'_ensemble.jsonl')
        mm={r['heterostructure_id']:r for r in rows(path)};assert set(mm)==set(ids)
        field=s['head_field'];out[task]=np.array([mm[i][field] for i in ids])
    out['ids']=ids
    return out

def calibration_diagnostics(rr,p):
    p=np.asarray(p);w=weights(rr);y=np.array([r['y'] for r in rr]);bins=[];ece=0.
    for lo,hi in zip(np.linspace(0,1,11)[:-1],np.linspace(0,1,11)[1:]):
        mask=(p>=lo)&((p<hi) if hi<1 else (p<=hi));mass=float(w[mask].sum())
        if not mask.any():continue
        a=float(np.dot(w[mask],p[mask])/mass);o=float(np.dot(w[mask],y[mask])/mass)
        bins.append({'bin_left':float(lo),'bin_right':float(hi),'mean_probability':a,'failure_frequency':o,'weight':mass,'rows':int(mask.sum())});ece+=mass*abs(a-o)
    return {'metrics':cm(rr,p),'equal_width_10bin_ECE':ece,'reliability_bins':bins}

def probability():
    target=ROOT/'calibration/probability.json'
    if target.exists():return read(target)
    b=read(ROOT/'base_selection_freeze.json');assert b['protocol_sha256']==sha(ROOT/'experiment_protocol.json')
    pred=selected_predictions('cal_probability');rr=load_role_labels('cal_probability')
    p=pred['classifier'];x=logit(np.clip(p,1e-6,1-1e-6));y=np.array([r['y'] for r in rr]);w=weights(rr,source_equal=False)
    groups=len({r['split_group_key'] for r in rr})
    def fg(theta):
        a,b=theta;z=a*x+b;q=expit(z)
        val=float(np.dot(w,np.logaddexp(0,z)-y*z)+1e-4*((a-1)**2+b*b))
        grad=np.array([np.dot(w,(q-y)*x)+2e-4*(a-1),np.dot(w,q-y)+2e-4*b])
        return val,grad
    if groups>=10 and len(set(y))==2:
        opt=minimize(fg,[1.,0.],jac=True,bounds=[(0,None),(None,None)],method='L-BFGS-B',options={'ftol':1e-14,'gtol':1e-10,'maxiter':2000})
        assert opt.success,opt.message;a,b=map(float,opt.x);fallback=False
    else:a,b=1.,0.;opt=None;fallback=True
    q=expit(a*x+b)
    out={'status':'FROZEN','created_utc':now(),'method':'global_monotone_Platt','a':a,'b':b,'input_clip':[1e-6,1-1e-6],
        'penalty':1e-4,'fit_weighting':'physical_material_pair_equal_no_source_reweighting','composition_groups':groups,'physical_pairs':len({r['material_pair_id'] for r in rr}),
        'rows':len(rr),'fallback_identity':fallback,'optimization':None if opt is None else {'success':bool(opt.success),'objective':float(opt.fun),'iterations':int(opt.nit),'gradient':opt.jac.tolist()},
        'base_selection_sha256':sha(ROOT/'base_selection_freeze.json'),'inference_manifest_sha256':sha(ROOT/'inference_manifest.json'),
        'role_ids_sha256':idsha(pred['ids']),'raw':calibration_diagnostics(rr,p),'calibrated':calibration_diagnostics(rr,q),
        'diagnostics_are_calibrator_fitting_data':True,'calibration_method_selection_performed':False}
    save(target,out)
    write_rows(ROOT/'calibration/probability_predictions.jsonl',[{'heterostructure_id':i,'p_raw':float(v),'p_calibrated':float(z),'y':r['y']} for i,v,z,r in zip(pred['ids'],p,q,rr)])
    print('Probability frozen',a,b,flush=True);return out

def apply_probability(p,cal):return expit(cal['a']*logit(np.clip(p,*cal['input_clip']))+cal['b'])

def fixed_risk(rr,p,policy):
    a=accepted(p,policy);w=weights(rr);y=np.array([r['y'] for r in rr]);m=float(w[a].sum())
    return {'weighted_coverage':m,'empirical_UAR':float(np.dot(w[a],y[a])/m) if m>0 else None,
        'accepted_records':int(a.sum()),'accepted_failure_records':int(y[a].sum()),'accepted_composition_groups':len({r['split_group_key'] for r,x in zip(rr,a) if x})}

def bootstrap_risk(rr,p,policy):
    groups=sorted({r['split_group_key'] for r in rr});indices={g:[i for i,r in enumerate(rr) if r['split_group_key']==g] for g in groups}
    rng=np.random.default_rng(20261025);out=[]
    for rep in range(1000):
        ix=[];sample=[]
        for k,g in enumerate(rng.choice(groups,size=len(groups),replace=True)):
            for i in indices[g]:
                ix.append(i);r=dict(rr[i]);r['material_pair_id']=r['material_pair_id']+f'|bootstrap_copy{k}';sample.append(r)
        out.append({'replicate':rep,**fixed_risk(sample,np.asarray(p)[ix],policy)})
    risks=[r['empirical_UAR'] for r in out if r['empirical_UAR'] is not None];cover=[r['weighted_coverage'] for r in out]
    point=fixed_risk(rr,p,policy)
    return out,{'replicates':1000,'seed':20261025,'unit':'global composition group, synchronized across sources, multiplicity preserved as distinct pair copies',
        'threshold_reselected_in_bootstrap':False,'percentile_95_UAR':np.quantile(risks,[.025,.975]).tolist() if risks else None,
        'percentile_95_coverage':np.quantile(cover,[.025,.975]).tolist(),'nonempty_risk_replicates':len(risks),
        'zero_event_degeneracy':point['accepted_failure_records']==0,'post_selection_intervals_are_descriptive':True,'risk_certificate_claim':False}

def decision():
    target=ROOT/'calibration/decision.json'
    if target.exists():return read(target)
    cal=read(ROOT/'calibration/probability.json');pred=selected_predictions('cal_decision');rr=load_role_labels('cal_decision')
    p=apply_probability(pred['classifier'],cal);cc=curve(rr,p);policy=choose_threshold(cc)
    boot,summary=bootstrap_risk(rr,p,policy)
    bas=np.array([r['g_A_eV'] for r in rr]);rg=np.maximum(0,bas+pred['residual_regression']);dg=np.maximum(0,pred['direct_gap_regression']);a=accepted(p,policy)
    routed=np.where(a,rg,dg)
    source_support={s:{'rows':sum(r['dataset']==s for r in rr),'physical_pairs':len({r['material_pair_id'] for r in rr if r['dataset']==s})} for s in sorted({r['dataset'] for r in rr})}
    out={'status':'FROZEN','created_utc':now(),'probability_sha256':sha(ROOT/'calibration/probability.json'),'base_selection_sha256':sha(ROOT/'base_selection_freeze.json'),
        'role_ids_sha256':idsha(pred['ids']),'rows':len(rr),'composition_groups':len({r['split_group_key'] for r in rr}),'source_support':source_support,
        'target_empirical_UAR':.05,'minimum_accepted_composition_groups':10,'policy':policy,'bootstrap':summary,
        'uncalibrated':calibration_diagnostics(rr,pred['classifier']),'calibrated':calibration_diagnostics(rr,p),
        'at_t0_5':fixed_risk(rr,p,{'policy':'threshold','threshold':.5}),
        'regression_comparisons':{k:rm(rr,v) for k,v in {'full_domain_residual':rg,'full_domain_direct':dg,'classifier_routed':routed,'raw_Anderson':bas}.items()},
        'regression_values_are_diagnostic_only':True,'service_changed_using_calibration':False,'risk_certificate_claim':False}
    write_rows(ROOT/'calibration/risk_coverage.jsonl',cc);write_rows(ROOT/'calibration/decision_bootstrap.jsonl',boot)
    write_rows(ROOT/'calibration/decision_predictions.jsonl',[{'heterostructure_id':i,'p_calibrated':float(z),'y':r['y'],'accepted':bool(ok),'residual_gap_eV':float(rg[k]),'direct_gap_eV':float(dg[k]),'routed_gap_eV':float(routed[k])} for k,(i,z,r,ok) in enumerate(zip(pred['ids'],p,rr,a))])
    save(target,out);print('Decision frozen',policy,flush=True);return out

def interval():
    target=ROOT/'calibration/interval.json'
    if target.exists():return read(target)
    dec=read(ROOT/'calibration/decision.json');cal=read(ROOT/'calibration/probability.json');b=read(ROOT/'base_selection_freeze.json')
    pred=selected_predictions('cal_interval');rr=load_role_labels('cal_interval');p=apply_probability(pred['classifier'],cal)
    gA=np.array([r['g_A_eV'] for r in rr]);rg=np.maximum(0,gA+pred['residual_regression']);dg=np.maximum(0,pred['direct_gap_regression']);routed=np.where(accepted(p,dec['policy']),rg,dg)
    gaps={'full_domain_residual':rg,'full_domain_direct':dg,'classifier_routed':routed};chosen={b['service'], 'classifier_routed'}
    out={'status':'FROZEN','created_utc':now(),'decision_sha256':sha(ROOT/'calibration/decision.json'),'probability_sha256':sha(ROOT/'calibration/probability.json'),
        'role_ids_sha256':idsha(pred['ids']),'rows':len(rr),'alpha':.1,'nominal_coverage':.9,'unit':'new composition group finite configuration set','service':b['service'],
        'claim':'nominal under composition-group exchangeability; source/track coverage to be evaluated in Step12','pipelines':{}}
    records=[]
    for name in sorted(chosen):
        v=gaps[name];scores={}
        for i,r in enumerate(rr):scores[r['split_group_key']]=max(scores.get(r['split_group_key'],0),abs(v[i]-r['g_PBE_eV']))
        G=len(scores);k=int(np.ceil((G+1)*.9));q=float(sorted(scores.values())[k-1]) if k<=G else None
        out['pipelines'][name]={'composition_groups':G,'quantile_index_1based':k,'q_eV':q,'unbounded':q is None,
            'score':'composition-group maximum absolute nonnegative gap error','group_scores_eV':scores,'calibration_group_coverage':sum(z<=q for z in scores.values())/G if q is not None else 1.,
            'calibration_row_coverage':float(np.mean(np.abs(v-np.array([r['g_PBE_eV'] for r in rr]))<=q)) if q is not None else 1.}
        for i,r in enumerate(rr):records.append({'pipeline':name,'heterostructure_id':pred['ids'][i],'split_group_key':r['split_group_key'],'predicted_gap_eV':float(v[i]),'g_PBE_eV':r['g_PBE_eV'],
             'lower_eV':max(0,float(v[i])-q) if q is not None else 0.,'upper_eV':float(v[i])+q if q is not None else None})
    write_rows(ROOT/'calibration/interval_predictions.jsonl',records);save(target,out)
    print('Interval frozen',{k:v['q_eV'] for k,v in out['pipelines'].items()},flush=True);return out

def main():
    assert read(ROOT/'inference_manifest.json')['status']=='PASS'
    probability();decision();interval()

if __name__=='__main__':main()
