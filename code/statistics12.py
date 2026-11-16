"""Array-only fixed-model metrics and synchronized composition bootstrap."""
from common12 import *

class GroupBootstrap:
    def __init__(self,rr,replicates=1000):
        self.rr=rr;self.n=len(rr);self.groups=sorted({r['split_group_key'] for r in rr});self.gindex={g:i for i,g in enumerate(self.groups)}
        self.rowgroup=np.array([self.gindex[r['split_group_key']] for r in rr]);self.source=np.array([r.get('source',r.get('dataset')) for r in rr])
        self.pair=np.array([r['material_pair_id'] for r in rr]);self.sources=sorted(set(self.source));self.base=weights(rr)
        rng=np.random.default_rng(20261025);G=len(self.groups)
        self.mult=rng.multinomial(G,np.full(G,1/G),size=replicates).astype(float);self.rowmult=self.mult[:,self.rowgroup]
        # Resampling synchronized global groups preserves all configurations and
        # duplicate draws, while allocating equal mass to physical-pair copies.
        self.raw=np.zeros_like(self.rowmult)
        for source in self.sources:
            ix=np.flatnonzero(self.source==source);pairs=sorted(set(self.pair[ix]));cnt={p:sum(self.pair[ix]==p) for p in pairs}
            den=np.zeros(replicates)
            for p in pairs:
                sub=ix[self.pair[ix]==p];assert len(set(self.rowgroup[sub]))==1
                den+=self.rowmult[:,sub[0]]
            self.raw[:,ix]=np.divide(self.rowmult[:,ix],den[:,None],out=np.zeros((replicates,len(ix))),where=den[:,None]>0)/np.array([cnt[p] for p in self.pair[ix]])
        active=np.zeros(replicates)
        for source in self.sources:active+=(self.raw[:,self.source==source].sum(axis=1)>0)
        self.w=np.divide(self.raw,active[:,None],out=np.zeros_like(self.raw),where=active[:,None]>0)
        self.replicates=replicates
    def ci(self,values):
        values=np.asarray(values,float);valid=values[np.isfinite(values)]
        return {'percentile_95':np.quantile(valid,[.025,.975]).tolist() if len(valid) else None,'defined_replicates':len(valid),'total_replicates':self.replicates}
    def mean(self,v,mask=None):
        v=np.asarray(v,float);mask=np.ones(self.n,bool) if mask is None else np.asarray(mask,bool)
        m=mask.astype(float);den=self.w@m
        return np.divide(self.w@np.where(mask,v,0),den,out=np.full(self.replicates,np.nan),where=den>0)
    def regression(self,true,pred):
        true,pred=np.asarray(true,float),np.asarray(pred,float);mask=np.isfinite(pred);err=pred-true
        mae=self.mean(np.abs(err),mask);mse=self.mean(err**2,mask)
        per=[]
        for s in self.sources:
            a=(self.source==s)&mask;den=self.raw@a.astype(float)
            z=np.divide(self.raw@np.where(a,err**2,0),den,out=np.full(self.replicates,np.nan),where=den>0);per.append(np.sqrt(z))
        ar=np.asarray(per);n=np.isfinite(ar).sum(axis=0);macro=np.divide(np.nansum(ar,axis=0),n,out=np.full(self.replicates,np.nan),where=n>0)
        return {'mae_eV':self.ci(mae),'rmse_eV':self.ci(macro),'rmse_eV_pooled_weighted':self.ci(np.sqrt(mse)),'bias_eV':self.ci(self.mean(err,mask))}
    def risk(self,y,accepted,processed):
        y=np.asarray(y,float);a=np.asarray(accepted,bool)&np.asarray(processed,bool)
        mass=self.w@a.astype(float);risk=np.divide(self.w@(a*y),mass,out=np.full(self.replicates,np.nan),where=mass>0)
        return {'UAR':self.ci(risk),'accepted_coverage':self.ci(mass),'processing_coverage':self.ci(self.w@np.asarray(processed,float))}
    def classifier(self,y,p,fail):
        y,p,fail=np.asarray(y,float),np.asarray(p,float),np.asarray(fail,bool);ok=np.isfinite(p);eps=1e-15;pc=np.clip(p,eps,1-eps)
        result={'brier':self.ci(self.mean((p-y)**2,ok)),'accuracy':self.ci(self.mean((fail==y).astype(float),ok)),
           'log_loss':self.ci(self.mean(-(y*np.log(pc)+(1-y)*np.log1p(-pc)),ok))}
        tp=self.w@(ok&(y==1)&fail);fn=self.w@(ok&(y==1)&~fail);fp=self.w@(ok&(y==0)&fail);tn=self.w@(ok&(y==0)&~fail)
        def ratio(a,b):return np.divide(a,b,out=np.full(self.replicates,np.nan),where=b>0)
        recall=ratio(tp,tp+fn);specificity=ratio(tn,tn+fp)
        result.update(pooled_failure_recall=self.ci(recall),pooled_precision=self.ci(ratio(tp,tp+fp)),pooled_balanced_accuracy=self.ci((recall+specificity)/2))
        ap_by=[];auc_by=[]
        for s in self.sources:
            ix=np.flatnonzero((self.source==s)&ok)
            if not len(ix):continue
            w=self.raw[:,ix];ys=y[ix];ps=p[ix]
            order=np.argsort(-ps,kind='stable');w=w[:,order];ys=ys[order];ps=ps[order]
            ends=np.r_[np.flatnonzero(ps[:-1]!=ps[1:]),len(ps)-1]
            tp=np.cumsum(w*ys,axis=1)[:,ends];mass=np.cumsum(w,axis=1)[:,ends]
            precision=np.divide(tp,mass,out=np.zeros_like(tp),where=mass>0)
            delta=np.diff(np.c_[np.zeros(len(w)),tp],axis=1);positive=tp[:,-1]
            ap=np.divide(np.sum(precision*delta,axis=1),positive,out=np.full(len(w),np.nan),where=positive>0);ap_by.append(ap)
            # Weighted pairwise concordance using low-to-high score groups.
            order=np.argsort(ps,kind='stable');p2=ps[order];w2=w[:,order];y2=ys[order]
            starts=np.r_[0,np.flatnonzero(p2[:-1]!=p2[1:])+1]
            pos=np.add.reduceat(w2*y2,starts,axis=1);neg=np.add.reduceat(w2*(1-y2),starts,axis=1)
            before=np.cumsum(neg,axis=1)-neg;den=pos.sum(axis=1)*neg.sum(axis=1)
            auc=np.divide(np.sum(pos*(before+.5*neg),axis=1),den,out=np.full(len(w),np.nan),where=den>0);auc_by.append(auc)
        for name,batch in [('source_macro_AP',ap_by),('source_macro_AUROC',auc_by)]:
            a=np.asarray(batch);count=np.isfinite(a).sum(axis=0);vals=np.divide(np.nansum(a,axis=0),count,out=np.full(self.replicates,np.nan),where=count>0);result[name]=self.ci(vals)
            if name=='source_macro_AP':self.last_AP_replicates=vals.copy()
        return result

def operational(rr,p,processed,t):
    p=np.asarray(p,float);processed=np.asarray(processed,bool);y=np.array([r['y'] for r in rr]);w=weights(rr);a=processed&(p<=t);f=processed&~a
    confusion={k:int(mask.sum()) for k,mask in {'tn':(y==0)&a,'fp':(y==0)&f,'fn':(y==1)&a,'tp':(y==1)&f}.items()}
    m=float(w[a].sum());row_risk=float(y[a].mean()) if a.any() else None
    return {'rows':len(rr),'processed_rows':int(processed.sum()),'abstained_rows':int((~processed).sum()),
      'row_processing_coverage':float(processed.mean()),'weighted_processing_coverage':float(w[processed].sum()),
      'accepted_rows':int(a.sum()),'accepted_failure_rows':int(y[a].sum()),'accepted_composition_groups':len({r['split_group_key'] for r,b in zip(rr,a) if b}),
      'row_accepted_coverage':float(a.mean()),'weighted_accepted_coverage':m,'row_UAR':row_risk,'weighted_UAR':float(w[a]@y[a]/m) if m else None,
      'confusion_row_counts_at_frozen_t':confusion,'abstention_by_true_class':{str(k):int(((y==k)&~processed).sum()) for k in (0,1)},
      'threshold':t,'risk_certificate_claim':False}

def interval_metrics(rr,pred,lo,hi,processed):
    y=np.array([r['g_PBE_eV'] for r in rr]);p=np.asarray(processed,bool);lo=np.array(lo,float);hi=np.array(hi,float);hit=p&(y>=lo)&(y<=hi);w=weights(rr)
    groups={g:[i for i,r in enumerate(rr) if r['split_group_key']==g] for g in sorted({r['split_group_key'] for r in rr})}
    full={g:bool(p[ix].all()) for g,ix in groups.items()};cover={g:bool(hit[ix].all()) for g,ix in groups.items()};G=len(groups);F=sum(full.values())
    return {'rows':len(rr),'row_unconditional_coverage':float(hit.mean()),'row_conditional_coverage':float(hit[p].mean()) if p.any() else None,
       'weighted_unconditional_row_coverage':float(w@hit),'weighted_conditional_row_coverage':float(w@hit/w[p].sum()) if p.any() else None,
       'mean_width_eV':float(np.mean((hi-lo)[p])) if p.any() else None,'composition_groups':G,'fully_processed_groups':F,
       'covered_groups':sum(cover.values()),'group_processing_coverage':F/G,'group_unconditional_coverage':sum(cover.values())/G,
       'group_conditional_coverage':sum(cover.values())/F if F else None,'group_covered':cover,'group_fully_processed':full}

def reliability(rr,p):
    p=np.asarray(p,float);ok=np.isfinite(p);y=np.array([r['y'] for r in rr]);w=weights(rr);out=[];ece=0.
    for j in range(10):
        a=ok&(p>=j/10)&((p<(j+1)/10) if j<9 else(p<=1));mass=w[a].sum()
        if not a.any():continue
        mean=float(w[a]@p[a]/mass);freq=float(w[a]@y[a]/mass);ece+=mass*abs(mean-freq)
        out.append({'bin':j,'left':j/10,'right':(j+1)/10,'rows':int(a.sum()),'weighted_mass':float(mass),'mean_probability':mean,'observed_failure':freq})
    return {'bins':out,'ECE10_conditional':float(ece/w[ok].sum()) if ok.any() else None}
