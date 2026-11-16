"""Frozen holdout evaluation. No fitting or test-based selection API exists."""
from statistics12 import *

def valid_regression(rr,pred,ci=True):
    pred=np.array(pred,float);mask=np.isfinite(pred);sub=[r for r,b in zip(rr,mask) if b]
    if not sub:return {'processed_rows':0,'metrics':None,'CI':None}
    metrics=rm(sub,pred[mask]);boot=GroupBootstrap(sub) if ci else None
    return {'processed_rows':len(sub),'metrics':metrics,'CI':boot.regression([r['g_PBE_eV'] for r in sub],pred[mask]) if ci else None}

def service_analysis(rr,primary,ci=True):
    n=len(rr);ok=np.array([p['prediction_status']=='in_domain' for p in primary]);p=np.array([v.get('p_anderson_failure') if b else np.nan for v,b in zip(primary,ok)],float)
    gap=np.array([v.get('predicted_PBE_gap_eV') if b else np.nan for v,b in zip(primary,ok)],float)
    dg=np.array([v.get('direct_branch_gap_eV') if b else np.nan for v,b in zip(primary,ok)],float)
    rg=np.array([v.get('routed_comparator',{}).get('predicted_PBE_gap_eV') if b else np.nan for v,b in zip(primary,ok)],float)
    t=read(S11/'calibration/decision.json')['policy']['threshold'];assert t is not None
    risk=operational(rr,p,ok,t);boot=GroupBootstrap(rr) if ci else None
    if ci:risk['CI']=boot.risk([r['y'] for r in rr],p<=t,ok)
    rows_valid=[r for r,b in zip(rr,ok) if b];classifier=None
    if rows_valid:
        pv=p[ok];hard=(pv>t).astype(float);hard_metrics=cm(rows_valid,hard)
        classifier={'calibrated':cm(rows_valid,pv),'raw':cm(rows_valid,[v['raw_p_anderson_failure'] for v,b in zip(primary,ok) if b]),
          'decision_at_frozen_t':{k:hard_metrics[k] for k in ('source_macro','by_source','pooled_source_pair_weighted','pooled_row_equal')},
          'reliability':reliability(rows_valid,pv),'raw_reliability':reliability(rows_valid,[v['raw_p_anderson_failure'] for v,b in zip(primary,ok) if b]),'CI':GroupBootstrap(rows_valid).classifier([r['y'] for r in rows_valid],pv,hard) if ci else None,
          'hard_decision_metrics_note':'Use accuracy, balanced accuracy, recall, precision, F1 and confusion from hard-decision block; its hard-score AP/Brier/logloss are not probability evaluation'}
    regressions={'service':valid_regression(rr,gap,ci),'routed':valid_regression(rr,rg,ci),'direct':valid_regression(rr,dg,ci),
       'Anderson_on_service_cohort':valid_regression(rr,np.where(ok,[r['g_A_eV'] for r in rr],np.nan),ci)}
    qs=read(S11/'calibration/interval.json')['pipelines'];intervals={}
    for name,v in [('service',gap),('routed',rg)]:
        q=qs['full_domain_residual' if name=='service' else 'classifier_routed']['q_eV'];assert q is not None
        lo=np.maximum(0,v-q);hi=v+q;z=interval_metrics(rr,v,lo,hi,ok)
        if ci:
            hit=ok&(np.array([r['g_PBE_eV'] for r in rr])>=lo)&(np.array([r['g_PBE_eV'] for r in rr])<=hi)
            cv=np.array([z['group_covered'][g] for g in boot.groups]);fp=np.array([z['group_fully_processed'][g] for g in boot.groups]);den=boot.mult@fp
            z['CI']={'weighted_unconditional_row':boot.ci(boot.mean(hit.astype(float))),
                'weighted_conditional_row':boot.ci(boot.mean(hit.astype(float),ok)),
                'unconditional_group':boot.ci(boot.mult@cv/len(boot.groups)),
                'conditional_group':boot.ci(np.divide(boot.mult@cv,den,out=np.full(1000,np.nan),where=den>0))}
        intervals[name]=z
    diffs={}
    if ok.any() and ci:
        bs=GroupBootstrap(rows_valid);y=np.array([r['g_PBE_eV'] for r in rows_valid]);e=np.abs(gap[ok]-y);base=np.abs(np.array([r['g_A_eV'] for r in rows_valid])-y);er=np.abs(rg[ok]-y)
        diffs={'service_minus_Anderson_MAE_eV':{'point':float(weights(rows_valid)@(e-base)),**bs.ci(bs.mean(e-base))},
           'routed_minus_service_MAE_eV':{'point':float(weights(rows_valid)@(er-e)),**bs.ci(bs.mean(er-e))}}
    return {'rows':n,'composition_groups':len({r['split_group_key'] for r in rr}),'physical_pairs':len({r['material_pair_id'] for r in rr}),
       'failure_prevalence_row':float(np.mean([r['y'] for r in rr])),'status_counts':{s:sum(v['prediction_status']==s for v in primary) for s in sorted({v['prediction_status'] for v in primary})},
       'risk':risk,'classifier':classifier,'regression':regressions,'interval':intervals,'paired_differences':diffs}

def curves(rr,primary):
    valid=np.array([r['prediction_status']=='in_domain' for r in primary]);p=np.array([r['p_anderson_failure'] if b else np.nan for r,b in zip(primary,valid)]);w=weights(rr);y=np.array([r['y'] for r in rr]);out=[]
    for t in np.unique(p[valid]):
        a=valid&(p<=t);mass=w[a].sum();out.append({'threshold':float(t),'weighted_coverage':float(mass),'weighted_UAR':float(w[a]@y[a]/mass) if mass else None,
            'row_coverage':float(a.mean()),'row_UAR':float(y[a].mean()) if a.any() else None,'accepted_rows':int(a.sum()),'failures':int(y[a].sum())})
    return out
