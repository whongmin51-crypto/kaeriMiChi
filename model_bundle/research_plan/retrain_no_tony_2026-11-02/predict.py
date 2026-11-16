"""New calibrated predictor; inherits original input validation, replaces learned state."""
from common import *
for k in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','BLIS_NUM_THREADS','OMP_NUM_THREADS'):os.environ[k]='1'
import numpy as np
import torch
old=module('original_frozen_api',S11/'freeze_predict.py')
old.VERSION='C1-no-Tony-fixed-config-2026-11-02'

class Predictor(old.FrozenPredictor):
    def __init__(self):
        self.contract=read(S11/'prediction_contract.json')
        self.contract['model_version']=old.VERSION
        self.contract['reference_tracks']=['native_free']
        self.contract['source_and_track_claim']='Janus/HetDB native_free fitting; performance evaluated in retrain_no_tony_2026-11-02/evaluation'
        self.state=read(ROOT/'inputs/fit_core_preprocessor.json')
        self.base=read(ROOT/'base_selection_freeze.json')
        self.probability=read(ROOT/'calibration/probability.json')
        self.decision=read(ROOT/'calibration/decision.json')
        self.interval=read(ROOT/'calibration/interval.json')
        self.preprocess=old.load_preprocessing()
        sys.path.insert(0,str(S10))
        from graph_engine import predict_arrays,set_seed
        from models import build_model
        set_seed(20261015);self.predict_arrays=predict_arrays;self.build_model=build_model;self.loaded={}
        self.jobs=read(ROOT/'jobs.json')
        self.contract_path=ROOT/'prediction_contract.json'
        self.contract['selected_models']={task:[{'path':(jobdir(j)/'best.pt').relative_to(RP).as_posix(),'sha256':read(jobdir(j)/'receipt.json')['checkpoint_sha256']} for j in self.jobs if j['candidate']==sel['candidate']] for task,sel in self.base['heads'].items()}
        self.contract['bindings_sha256']={p.relative_to(RP).as_posix():sha(p) for p in [ROOT/'protocol.json',ROOT/'inputs/fit_core_preprocessor.json',ROOT/'calibration/probability.json',ROOT/'calibration/decision.json',ROOT/'calibration/interval.json',Path(__file__),S10/'models.py',S10/'graph_engine.py']}
        self.contract['preprocessing_state_path']=(ROOT/'inputs/fit_core_preprocessor.json').relative_to(RP).as_posix()
        self.contract['preprocessing_state_sha256']=sha(ROOT/'inputs/fit_core_preprocessor.json')
        save(self.contract_path,self.contract)
    def predict(self,request,request_base=ROOT):
        if isinstance(request,dict) and request.get('protocol',{}).get('reference_track')=='matched_strain':
            out=self._base_output(request.get('record_id'));out['errors']=['This two-source fitted release supports native_free references only.'];return out
        return super().predict(request,request_base=request_base)
    def _heads(self,adapter):
        output={};internal={'classifier':'classifier','residual_regression':'residual','direct_gap_regression':'direct_gap'}
        for task,selected in self.base['heads'].items():
            vals=[]
            for j in self.jobs:
                if j['candidate']!=selected['candidate']:continue
                key=(j['candidate'],j['seed']);out=jobdir(j)
                if key not in self.loaded:
                    r=read(out/'receipt.json');assert sha(out/'best.pt')==r['checkpoint_sha256']
                    cp=torch.load(out/'best.pt',map_location='cpu',weights_only=False);s=cp['specification']
                    assert s['preprocessor_sha256']==sha(ROOT/'inputs/fit_core_preprocessor.json')
                    model=self.build_model(s['family'],s['params'],n_numeric=1540,tasks=s['tasks']);model.load_state_dict(cp['model_state_dict']);model.eval()
                    self.loaded[key]=(model,tuple(s['tasks']))
                model,tasks=self.loaded[key];p=self.predict_arrays(model,adapter,['new_input'],'fit_core',tasks);vals.append(float(p[internal[task]][0]))
            assert len(vals)==5;output[task]=float(np.mean(vals))
        return output

if __name__=='__main__':
    import argparse
    ap=argparse.ArgumentParser();ap.add_argument('--request',type=Path);a=ap.parse_args();pr=Predictor()
    if a.request:print(json.dumps(pr.predict(read(a.request),request_base=a.request.parent),ensure_ascii=False))
    else:
        import copy
        root=RP/'step15_same_source_blind_2026-10-27';out=[]
        for path in sorted((root/'geometry/bilayers').glob('B*/model_request.json')):
            request=read(path);prediction=pr.predict(request,request_base=path.parent)
            out.append({'calculation_id':request['record_id'],'prediction':prediction,'input_sha256':sha(path),'evaluation_design':'retrospective evaluation against existing DFT; no new blind-test claim'})
            print(request['record_id'],prediction['prediction_status'],prediction.get('predicted_PBE_gap_eV'),flush=True)
        write_rows(ROOT/'application/retrospective_predictions.jsonl',out)
        tests=[];path=next((root/'geometry/bilayers').glob('B01/model_request.json'))
        for name in ('missing_band_edge','SOC_mismatch','true_twist'):
            req=copy.deepcopy(read(path))
            if name=='missing_band_edge':req['raw_features']['upper__vbm_contact_eV']=None
            elif name=='SOC_mismatch':req['protocol']['soc']=True
            else:req['raw_features']['twist_sin']=.5;req['raw_features']['twist_cos']=.8660254
            pred=pr.predict(req,request_base=path.parent);assert pred.get('predicted_PBE_gap_eV') is None
            tests.append({'case':name,'prediction':pred})
        save(ROOT/'application/input_validation.json',{'status':'PASS','tests':tests})
