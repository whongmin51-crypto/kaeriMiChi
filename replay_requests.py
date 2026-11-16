"""Replay six archived native-reference requests through the calibrated service."""
from pathlib import Path
import copy
import hashlib
import json
import sys
import numpy as np
ROOT=Path(__file__).resolve().parent
BUNDLE=ROOT/'model_bundle'
RUN=BUNDLE/'research_plan/retrain_no_tony_2026-11-02'
sys.path.insert(0,str(RUN))
import predict as service


def read(path):return json.loads(Path(path).read_text(encoding='utf8'))


def main():
    out=ROOT/'replayed';out.mkdir(exist_ok=True)
    contract=out/'portable_prediction_contract.json'
    # The original service writes a derived contract on initialization. Redirect
    # only that write; the packaged scientific inputs/checkpoints remain immutable.
    service.save=lambda path,value:contract.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf8')
    predictor=service.Predictor();predictor.contract_path=contract
    expected={r['calculation_id']:r['prediction'] for r in
              [json.loads(line) for line in (ROOT/'data/application/retrospective_predictions.jsonl').read_text(encoding='utf8').splitlines() if line.strip()]}
    outputs=[]
    keys=['anderson_gap_eV','anderson_signed_gap_eV','p_anderson_failure','raw_p_anderson_failure',
          'predicted_residual_eV','predicted_PBE_gap_eV','prediction_interval_eV',
          'raw_residual_branch_gap_eV','raw_direct_gap_eV','residual_branch_gap_eV','direct_branch_gap_eV']
    for path in sorted((BUNDLE/'example_requests').glob('B*/request.json')):
        request=read(path);actual=predictor.predict(request,request_base=path.parent)
        old=expected[request['record_id']]
        assert actual['prediction_status']==old['prediction_status']=='in_domain'
        for key in keys:
            assert np.allclose(actual[key],old[key],atol=1e-7,rtol=1e-7),(request['record_id'],key,actual[key],old[key])
        outputs.append({'case':request['record_id'],'prediction':actual,'numeric_comparison':'PASS'})
    path=BUNDLE/'example_requests/B01/request.json';original=read(path)
    tests=[]
    for case in ['missing_band_edge','SOC_mismatch','true_twist','matched_strain']:
        request=copy.deepcopy(original)
        if case=='missing_band_edge':request['raw_features']['upper__vbm_contact_eV']=None
        elif case=='SOC_mismatch':request['protocol']['soc']=True
        elif case=='true_twist':request['raw_features']['twist_sin']=.5;request['raw_features']['twist_cos']=.8660254
        else:request['protocol']['reference_track']='matched_strain'
        actual=predictor.predict(request,request_base=path.parent)
        assert actual['predicted_PBE_gap_eV'] is None and actual['prediction_status']!='in_domain'
        tests.append({'case':case,'status':actual['prediction_status'],'errors':actual['errors']})
    result={'status':'PASS','requests_replayed':len(outputs),'invalid_or_unsupported_cases':tests,
            'scope':'six existing retrospective requests; raw feature values and NPZ graphs supplied',
            'raw_structure_feature_extraction_verified':False,
            'note':'Portable graph path spellings alter input/contract hashes; scientific values compare to the archived requests.'}
    (out/'request_predictions.json').write_text(json.dumps(outputs,indent=2,ensure_ascii=False),encoding='utf8')
    (out/'requests_validation.json').write_text(json.dumps(result,indent=2),encoding='utf8')
    print(json.dumps(result))


if __name__=='__main__':main()
