"""Build metadata-only three-input vectors; never read bilayer target labels."""
from pathlib import Path
from collections import Counter
from datetime import datetime,timezone
import json,hashlib,sys
import numpy as np
from input_features import pair_features,graph_intrinsics,schema,dense_values

ROOT=Path(__file__).resolve().parent;RP=ROOT.parent;S7=RP/'step07_2026-10-15'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def rows(p):return [json.loads(s) for s in Path(p).read_text(encoding='utf-8-sig').splitlines() if s]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def write_rows(p,data):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('w',encoding='utf-8') as f:
        for row in data:f.write(json.dumps(row,ensure_ascii=False,allow_nan=False,separators=(',',':'))+'\n')

def source_tables():
    mono={r['monolayer_reference_id']:r for r in rows(RP/'step03_2026-10-15/tables/monolayers.jsonl')}
    identity={r['monolayer_reference_id']:r for r in rows(RP/'step04_2026-10-15/materials/mono_identity_map.jsonl')}
    for key,m in mono.items():m['face_equivalent']=not identity[key]['phase_normal_orientation_identifiable']
    configs={r['configuration_id']:r for r in rows(RP/'step03_2026-10-15/tables/configurations.jsonl')}
    contexts={r['heterostructure_id']:r for r in rows(RP/'step05_2026-10-15/tables/baseline_context_manifest.jsonl')}
    return mono,configs,contexts

def payload_from_sources(context,config,monos,profile='available_initial_stacking'):
    """Only qian continuous geometry is admitted; external final geometry ignored."""
    layers=context['layers']
    payload={'input_profile':profile,'reference_track':context['reference_track'],
        'monolayers':{l:monos[layers[l]['monolayer_reference_id']] for l in 'AB'},
        'stacking':{'lower_layer':'A' if context['dataset']=='Janus' or layers['A'].get('bilayer_location')=='lower' else 'B',
                    'contact_side_A':layers['A']['contact_side'],'contact_side_B':layers['B']['contact_side'],
                    'continuous_stage':'unavailable',
                    'twist_angle_deg':None,'relative_motif_angle_mod120_deg':None,
                    'translation_uv':None,'initial_contact_distance_A':None,'strain_A':None,'strain_B':None}}
    if context['dataset']=='Janus':
        qian=next(s for s in config['stages'] if s['stage']=='qian')
        if not qian['predictive_input_eligible'] or qian['model_stage_role']!='input_geometry':raise ValueError('Qian input stage not admitted')
        qlayers={l['layer_index']:l for l in qian['layers']}
        for letter in 'AB':
            if qlayers[letter]['contact_side_of_current_reference']!=layers[letter]['contact_side']:raise ValueError('Qian face and frozen baseline disagree')
        payload['stacking'].update({'continuous_stage':'qian_candidate_input',
            'relative_motif_angle_mod120_deg':qian['relative_motif_angle_mod120_deg'],
            'translation_uv':[qian['translation_u_fraction'],qian['translation_v_fraction']],
            'initial_contact_distance_A':qian['layer_distance_A'],
            'strain_A':qlayers['A']['strain_vs_current_reference'],'strain_B':qlayers['B']['strain_vs_current_reference']})
    return payload

def main():
    if (ROOT/'validation.json').exists() and read(ROOT/'validation.json').get('completed'):raise RuntimeError('Step08 is published; create a new version for regeneration')
    gate=read(S7/'validation.json');assert gate['status']=='PASS' and gate['completed'] and gate['ready_for_step08']
    snapshot=ROOT/'input_manifest.json'
    if snapshot.exists():
        protected=read(snapshot)['immutable_inputs']
        for p,h in protected.items():assert sha(p)==h,p
    else:
        for p,h in gate['output_sha256'].items():assert sha(p)==h,p
        protected=dict(read(S7/'input_manifest.json')['immutable_inputs'])
        for p,h in gate['output_sha256'].items():
            if str(Path(p)).startswith(str(S7)) or str(Path(p)).startswith(str(RP/'reports')):protected[p]=h
        protected[str(S7/'validation.json')]=sha(S7/'validation.json')
        save(snapshot,{'created_utc':datetime.now(timezone.utc).isoformat(),'immutable_inputs':protected,
                       'sealed_test_access':'SHA only','step07_snapshot_sha256':sha(S7/'validation.json')})
    save(ROOT/'validation.json',{'status':'RUNNING','completed':False,'stage':8,'step09_ready':False})
    graph_validation=read(ROOT/'graphs/validation.json');assert graph_validation['status']=='PASS'
    mono,cfg,ctx=source_tables();part={r['heterostructure_id']:r for r in rows(S7/'record_partitions.jsonl')}
    graph_map={r['monolayer_reference_id']:r for r in rows(ROOT/'graphs/monolayer_graph_map.jsonl')}
    sch=schema();save(ROOT/'features/feature_schema.json',sch)
    intrinsics={key:graph_intrinsics(m) for key,m in mono.items()}
    feature_rows=[];availability=[];auxiliary=[];raw_matrix=[]
    for record_id,c in ctx.items():
        payload=payload_from_sources(c,cfg[c['configuration_id']],mono)
        inf={label:intrinsics[payload['monolayers'][label]['monolayer_reference_id']] for label in 'AB'}
        packet=pair_features(payload,inf);p=part[record_id];L=packet['physical_order_labels']['lower'];U=packet['physical_order_labels']['upper']
        graph_refs={pos:graph_map[payload['monolayers'][label]['monolayer_reference_id']]['graph_id'] for pos,label in [('lower',L),('upper',U)]}
        value_row={'heterostructure_id':record_id,'configuration_id':c['configuration_id'],'profile':packet['profile'],
            'values':packet['values'],'graph_ids':graph_refs,'reference_track':payload['reference_track']}
        feature_rows.append(value_row);raw_matrix.append(dense_values(packet['values'],sch))
        provenance={'electronic_sources':{l:{'monolayer_reference_id':payload['monolayers'][l]['monolayer_reference_id'],
            'source_sha256':payload['monolayers'][l]['source_sha256'],
            'structure_sha256':payload['monolayers'][l]['structure_sha256']} for l in 'AB'},
            'contact_face_origin':'qian_candidate_given_face' if c['dataset']=='Janus' else 'given_contact_condition_inferred_from_final_geometry_not_verified_original_design',
            'continuous_stack_origin':'qian_candidate_input' if c['dataset']=='Janus' else 'unavailable_not_filled_from_final_DFT',
            'reference_scope':'pair-specific isolated matched-strain inputs' if c['dataset']=='Tony' else 'native/free isolated monolayers',
            'source_qian_structure_sha256':next(s['structure_sha256'] for s in cfg[c['configuration_id']]['stages'] if s['stage']=='qian') if c['dataset']=='Janus' else None}
        availability.append({'heterostructure_id':record_id,'dataset':c['dataset'],'original_role':p['original_role'],'partition':p['partition'],
            'profile':'available_initial_stacking','label_role_eligible':p['eligible_for_assigned_role'],
            'conditional_input_eligible':True,'baseline_available':packet['baseline_available'],
            'complete_candidate_stacking_available':packet['stacking_complete'],
            'candidate_qian_stage_available':c['dataset']=='Janus',
            'historical_source_DFT_stage_independently_verified':False,
            'source_reported_twist_used':False,'final_DFT_continuous_geometry_used':False,
            'core_task_eligibility':{task:p['eligible_for_assigned_role'] and packet['baseline_available'] for task in ['classifier','residual_regression','direct_gap_regression']},
            'provenance':provenance})
    for key,m in mono.items():
        alpha=m.get('polarizability_z_A')
        if alpha is None and isinstance(m.get('polarizability_A'),dict):alpha=m['polarizability_A'].get('alphaz')
        auxiliary.append({'monolayer_reference_id':key,'dataset':m['dataset'],'main_input_enabled':False,
            'effective_mass_cb_me':m.get('effective_mass_cb_m_e',m.get('electron_effective_mass_me')),
            'effective_mass_vb_me':m.get('effective_mass_vb_m_e',m.get('hole_effective_mass_me')),
            'polarizability_z_A':alpha,'orbital_projection':m.get('orbital_projection'),
            'auxiliary_soc_status':m.get('auxiliary_soc_status',m.get('source_soc_status')),
            'reason':'optional auxiliary properties not aligned to common noSOC channel; Janus missing',
            'source_sha256':m['source_sha256']})
    write_rows(ROOT/'features/features.jsonl',feature_rows)
    write_rows(ROOT/'features/input_availability.jsonl',availability)
    write_rows(ROOT/'features/auxiliary_properties_not_in_main.jsonl',auxiliary)
    raw=np.stack(raw_matrix)
    np.savez_compressed(ROOT/'features/raw_feature_matrix.npz',record_ids=np.array([r['heterostructure_id'] for r in feature_rows]),
                        feature_names=np.array(sch['ordered_feature_names']),values=raw)
    # Save only non-test, non-calibration examples for actual interface verification.
    core_candidates=sorted((r for r in availability if r['partition']=='fit_core' and r['dataset']=='Janus'),key=lambda r:r['heterostructure_id'])
    polar=next(r for r in core_candidates if any(ctx[r['heterostructure_id']]['layers'][l]['contact_side']!='equivalent_low_high' for l in 'AB'))
    rid=polar['heterostructure_id'];example=payload_from_sources(ctx[rid],cfg[ctx[rid]['configuration_id']],mono)
    save(ROOT/'examples/core_polar_payload.json',{'heterostructure_id':rid,'payload':example,'example_scope':'fit_core input only; no target'})
    summary={'status':'PASS','source_records':len(feature_rows),'monolayer_references':len(mono),'raw_feature_count':len(sch['ordered_feature_names']),
        'profile':'available_initial_stacking','feature_file_sha256':sha(ROOT/'features/features.jsonl'),
        'feature_schema_sha256':sha(ROOT/'features/feature_schema.json'),
        'input_constructor_sha256':sha(ROOT/'input_features.py'),
        'known_zero_atomic_fractions_sparse':True,'source_reported_external_twist_used':False,
        'external_final_geometry_used_as_initial':False,'bilayer_target_labels_read':False,
        'source_role_counts':[{ 'dataset':d,'role':role,'records':sum(r['dataset']==d and r['original_role']==role for r in availability),
            'full_candidate_stack':sum(r['dataset']==d and r['original_role']==role and r['complete_candidate_stacking_available'] for r in availability)}
            for d in ['Janus','HetDB','Tony'] for role in ['train','validation','test','quarantine']
            if any(r['dataset']==d and r['original_role']==role for r in availability)],
        'finite_matrix_entries':int(np.isfinite(raw).sum()),'missing_matrix_entries':int(np.isnan(raw).sum()),
        'matrix_shape':list(raw.shape),'auxiliary_property_records':len(auxiliary)}
    save(ROOT/'features/validation.json',summary)
    for p,h in protected.items():assert sha(p)==h,p
    print(json.dumps(summary,ensure_ascii=False))

if __name__=='__main__':main()
