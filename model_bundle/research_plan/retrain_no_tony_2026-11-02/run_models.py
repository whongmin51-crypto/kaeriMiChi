from common import *
for k in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','BLIS_NUM_THREADS','OMP_NUM_THREADS'):os.environ[k]='1'
import argparse,time,gc,traceback
import numpy as np

class Inputs:
    def __init__(self, graph=False):
        protocol=read(ROOT/'protocol.json')
        for p,h in protocol['inputs_sha256'].items():assert sha(ROOT/p)==h,p
        self.roles=read(ROOT/'inputs/role_ids.json');self.fit=read(ROOT/'inputs/core_metadata.json')
        self.fitids=[r['heterostructure_id'] for r in self.fit]
        self.metadata={r['heterostructure_id']:r for r in self.fit}
        self.partition={r['heterostructure_id']:r for r in rows(ROOT/'inputs/partitions.jsonl')}
        with np.load(ROOT/'inputs/raw_features.npz',allow_pickle=False) as ar:
            self.ids=ar['record_ids'].astype(str).tolist();self.raw=ar['values'];self.names=ar['feature_names'].astype(str).tolist()
        self.index={k:i for i,k in enumerate(self.ids)}
        self.state_value=read(ROOT/'inputs/fit_core_preprocessor.json')
        pp=module('no_tony_pp',S8/'preprocessing/preprocessing.py');tr=pp.transform(self.raw,self.state_value)
        self.tree_values=tr['tree_values'].astype(np.float32)
        self.unsupported=tr['unsupported_feature_mask'].sum(axis=1)
        self.g_A=self.raw[:,self.names.index('anderson_nonnegative_eV')]
        self.states={'fit_core':{'sha256':sha(ROOT/'inputs/fit_core_preprocessor.json')}}
        self.input_binding={'protocol_sha256':sha(ROOT/'protocol.json'),'adapter_sha256':sha(Path(__file__)),
           'inputs_sha256':protocol['inputs_sha256'],'calibration_external_test_labels_read':False,'source_filter':'dataset != Tony'}
        if graph:
            import torch
            self.graph_values=torch.from_numpy(tr['graph_values'].astype(np.float32));self.graph_pairs=read(ROOT/'inputs/graph_pairs.json');self.graph_cache={}
            for r in read(ROOT/'inputs/graph_manifest.json'):
                p=S8/'graphs/cache'/(r['graph_id'].split(':',1)[1]+'.npz');assert sha(p)==r['sha256']
                with np.load(p,allow_pickle=False) as a:
                    self.graph_cache[r['graph_id']]={k:torch.from_numpy(a[k].astype(np.int64 if k in ('atomic_numbers','edge_index') else np.float32)) for k in ('atomic_numbers','edge_index','edge_rbf','edge_vectors_A')}
    def state(self,scope):assert scope=='fit_core';return self.state_value
    def numeric(self,scope):assert scope=='fit_core';return self.graph_values
    def validate_ids(self,ids):assert ids and len(ids)==len(set(ids)) and set(ids)<=set(self.ids)
    def metadata_for(self,ids):assert set(ids)<=set(self.fitids);return [self.metadata[k] for k in ids]
    def dataset(self,scope,ids):
        from graph_data import PairDataset
        assert set(ids)<=set(self.fitids);return PairDataset(self,scope,ids)

def run(backend,shard,shards):
    ledgerpath=ROOT/f'progress/{backend}_{shard}.json'
    selected=[j for j in read(ROOT/'jobs.json') if j['backend']==backend][shard::shards]
    data=Inputs(graph=backend=='graph');ledger={'status':'RUNNING','pid':os.getpid(),'started_utc':now(),'total':len(selected),'completed':0,'events':[]}
    save(ledgerpath,ledger)
    if backend=='graph':
        sys.path.insert(0,str(S10));from graph_engine import fit_model,predict_model
    else:engine=module('no_tony_tree_engine',S9/'engine.py')
    predids=[k for role in ROLES[1:] for k in data.roles[role]]
    for j in selected:
        verify_job(j);out=jobdir(j);out.mkdir(parents=True,exist_ok=True)
        ledger.update(current=j['candidate'],seed=j['seed'],current_started_utc=now());save(ledgerpath,ledger)
        receiptpath=out/'receipt.json'
        if receiptpath.exists():
            receipt=read(receiptpath);assert receipt['status']=='PASS' and sha(out/receipt['checkpoint'])==receipt['checkpoint_sha256']
            assert sha(out/'predictions.npz')==receipt['predictions_sha256']
        else:
            start=time.perf_counter()
            if backend=='graph':
                result=fit_model(data,j['family'],j['params'],j['internal_task'],data.fitids,'fit_core',out,j['seed'],stop_ids=None,epochs=j['iterations'],patience=j['patience'],loss_weights=j['loss_weights'])
                pred=predict_model(data,out,predids);pred.pop('ids');checkpoint='best.pt';duration=result['training_seconds']
                spec=read(out/'fit_specification.json')
                assert spec['params']==j['params'] and spec['loss_weights']==j['loss_weights'] and spec['epochs_cap']==j['iterations']
                assert spec['n_fit']==11561 and spec['n_stop']==0 and result['last_epoch']==j['iterations']
            else:
                ix=[data.index[k] for k in data.fitids];meta=data.fit
                target=np.array([r[{'classifier':'y','residual_regression':'r_eV','direct_gap_regression':'g_PBE_eV'}[j['task']]] for r in meta])
                checkpoint='model.joblib' if j['family']=='RF' else 'model.json'
                result=engine.fit_model(j['family'],j['task'],j['params'],j['seed'],data.tree_values[ix],target,
                  g_A=data.g_A[ix],source=np.array([r['dataset'] for r in meta]),pair=np.array([r['material_pair_id'] for r in meta]),early_stop=None,iterations=j['iterations'],output_path=out/checkpoint)
                fitted=engine.load_model(out/checkpoint,j['family'])
                values=engine.predict_head(fitted,j['family'],j['task'],data.tree_values[[data.index[k] for k in predids]])
                pred={{'classifier':'classifier','residual_regression':'residual','direct_gap_regression':'direct_gap'}[j['task']]:values}
                duration=result.metadata['fit_seconds'];save(out/'fit_metadata.json',result.metadata);save(out/'curve.json',result.curve);save(out/'job.json',j)
                del fitted,result
            assert all(np.isfinite(v).all() for v in pred.values())
            np.savez_compressed(out/'predictions.npz',ids=np.array(predids),**pred)
            receipt={'status':'PASS','completed_utc':now(),'candidate':j['candidate'],'seed':j['seed'],'actual_new_fit':True,
              'warm_start_from_original_weights':False,'same_hyperparameters':True,'same_seed':True,'same_fixed_iterations':True,
              'fit_rows':11561,'fit_id_sha256':idsha(data.fitids),'source_counts':{'Janus':11530,'HetDB':31},
              'checkpoint':checkpoint,'checkpoint_sha256':sha(out/checkpoint),'predictions_sha256':sha(out/'predictions.npz'),
              'fit_seconds':duration,'runtime_seconds':time.perf_counter()-start,'prediction_rows':len(predids),
              'protocol_sha256':sha(ROOT/'protocol.json'),'original_checkpoint_preserved':sha(RP/j['original_checkpoint'])==j['original_sha256'],
              'test_and_calibration_labels_used_for_fit':False}
            assert receipt['original_checkpoint_preserved'];save(receiptpath,receipt)
        ledger['completed']+=1;ledger['events'].append({'candidate':j['candidate'],'seed':j['seed'],'status':'PASS','fit_seconds':receipt['fit_seconds']});save(ledgerpath,ledger)
        print(f"PASS {j['candidate']} seed{j['seed']} ({ledger['completed']}/{len(selected)})",flush=True);gc.collect()
    ledger.update(status='PASS',completed_utc=now());save(ledgerpath,ledger)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--backend',choices=['trees','graph'],required=True);p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=1);a=p.parse_args()
    try:run(a.backend,a.shard,a.shards)
    except BaseException:
        save(ROOT/f'progress/{a.backend}_{a.shard}_failure.json',{'status':'FAIL','at_utc':now(),'traceback':traceback.format_exc()});raise
