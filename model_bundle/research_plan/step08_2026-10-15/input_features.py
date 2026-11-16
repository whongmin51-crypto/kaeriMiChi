"""Three-input contract: isolated monolayers, atom graphs and explicit stacking.

No bilayer target, final geometry, quality flag, dataset identity or split role
is read by the numeric feature constructor. A/B names are cosmetic; physical
lower/upper order is explicit. All energies are vacuum-referenced noSOC eV.
"""
from pathlib import Path
from collections import Counter
import hashlib,json,math,sys
import numpy as np

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'graphs'))
import graph_builder as gb

PROFILES=('available_initial_stacking','contact_conditioned')
STRAIN_FIELDS=('a_fraction','b_fraction','angle_change_deg','principal_max_fraction','principal_min_fraction',
               'area_change_fraction','green_xx','green_yy','green_xy')
MONO_SPECS={
 'gap_eV':('eV','gap','isolated-layer noSOC band-edge interval'),
 'vbm_average_eV':('eV','band_edges','VBM versus declared average/source vacuum'),
 'cbm_average_eV':('eV','band_edges','CBM versus declared average/source vacuum'),
 'midgap_average_eV':('eV','band_edges','mean of isolated VBM and CBM'),
 'vbm_contact_eV':('eV','contact_band_edges','VBM versus the given contact-face vacuum'),
 'cbm_contact_eV':('eV','contact_band_edges','CBM versus the given contact-face vacuum'),
 'midgap_contact_eV':('eV','contact_band_edges','mean contact-face edges'),
 'electron_affinity_contact_eV':('eV','contact_band_edges','negative contact CBM'),
 'ionization_energy_contact_eV':('eV','contact_band_edges','negative contact VBM'),
 'vacuum_step_stack_eV':('eV','polarity','high-minus-low vacuum step in physical stacking normal'),
 'dipole_density_stack_e_per_A':('e/Angstrom','polarity','isolated dipole component along stacking normal divided by cell area'),
 'lattice_a_A':('Angstrom','lattice','length of supplied reference a vector'),
 'lattice_b_A':('Angstrom','lattice','length of supplied reference b vector'),
 'lattice_gamma_deg':('deg','lattice','angle between reference a,b'),
 'area_per_atom_A2':('Angstrom^2','lattice','reference in-plane area divided by atom count'),
 'thickness_A':('Angstrom','thickness','highest minus lowest atom along a cross b'),
 'Z_mean':('1','composition_stats','mean atomic number'),
 'Z_std':('1','composition_stats','population standard deviation of atomic numbers'),
 'Z_min':('1','composition_stats','minimum atomic number'),
 'Z_max':('1','composition_stats','maximum atomic number'),
 'element_count':('1','composition_stats','number of distinct elements'),
 'neighbor_count_mean':('1','local_geometry','mean number of 2D periodic neighbors at fixed cutoff'),
 'neighbor_count_std':('1','local_geometry','standard deviation of neighbor counts'),
 'neighbor_distance_mean_A':('Angstrom','local_geometry','mean retained periodic edge distance'),
 'neighbor_distance_std_A':('Angstrom','local_geometry','standard deviation of retained periodic distances'),
 'contact_outer_Z_mean':('1','contact_chemistry','mean Z of outer atomic plane on given contact side'),
 'contact_adjacent_Z_mean':('1','contact_chemistry','mean Z of next atomic plane on given contact side'),
 'contact_side_canonical_code':('1','face_orientation','-1 low, +1 high relative to deterministic chemical plane sequence; 0 equivalent faces'),
 'flip_from_chemical_canonical':('1','face_orientation','physical normal reversal relative to chemical-canonical reference'),
 'face_equivalent':('1','face_orientation','verified or explicitly declared normal-reversal equivalent faces'),
}
OPS=('lower','upper','mean','signed_delta','abs_delta')
STACK_COLUMNS={
 'twist_sin':('1','stack_twist','sin of genuine user-specified upper-relative-lower twist; full 360 degree period'),
 'twist_cos':('1','stack_twist','cos of genuine user-specified twist'),
 'motif_sin3':('1','stack_registry','sin(3*motif angle), for the independent 120 degree motif convention'),
 'motif_cos3':('1','stack_registry','cos(3*motif angle), not a true twist angle'),
 'translation_u_sin':('1','stack_translation','sin(2 pi u) for upper-minus-lower contact anchor in common cell'),
 'translation_u_cos':('1','stack_translation','cos(2 pi u)'),
 'translation_v_sin':('1','stack_translation','sin(2 pi v)'),
 'translation_v_cos':('1','stack_translation','cos(2 pi v)'),
 'initial_contact_distance_A':('Angstrom','stack_distance','given initial nearest lower-high/upper-low atomic-plane distance'),
}

def finite(x):
    if x is None:return None
    if isinstance(x,(bool,np.bool_)):raise ValueError('A numeric input cannot be a boolean')
    try:value=float(x)
    except (TypeError,ValueError,OverflowError) as exc:raise ValueError('A numeric input must be finite or explicitly null') from exc
    if not math.isfinite(value):raise ValueError('Nonfinite input feature')
    return value

def required_finite(x,name):
    value=finite(x)
    if value is None:raise ValueError(name+' is required and must be finite')
    return value

def reject_explicit_soc(obj):
    """Reject a declared SOC input without treating unreported source evidence as SOC."""
    for key in ('experiment_soc','soc','soc_enabled','with_soc','include_soc','use_soc'):
        value=obj.get(key)
        if isinstance(value,(bool,np.bool_)):
            enabled=bool(value)
        elif isinstance(value,(int,float)):
            enabled=value==1
        elif isinstance(value,str):
            enabled=value.strip().lower() in ('true','yes','1','enabled','on','with_soc')
        else:enabled=False
        if enabled:raise ValueError('This input contract requires noSOC electronic properties: '+key)
    for key in ('electronic_channel','bandedge_channel','band_edge_channel','soc_channel'):
        value=obj.get(key)
        if isinstance(value,str) and value.strip().lower().replace('-','_') in (
                'soc','with_soc','gaps_soc','including_soc','spin_orbit','pbe_soc'):
            raise ValueError('This input contract requires a noSOC electronic channel')

def normalized_contact_side(m,side):
    if side not in ('low','high','equivalent_low_high'):raise ValueError('Explicit contact side is required')
    equivalent=m.get('face_equivalent',False)
    if not isinstance(equivalent,(bool,np.bool_)):raise ValueError('face_equivalent must be an explicit boolean')
    if equivalent:return 'equivalent_low_high'
    if side=='equivalent_low_high':raise ValueError('Equivalent faces require verified/explicit symmetry metadata')
    return side

def graph_intrinsics(m):
    g=gb.build_graph(m)
    z=np.asarray(g['atomic_numbers'],dtype=int);cell=np.asarray(m['cell_A'],float);pos=np.asarray(m['positions_A'],float)
    normal=np.cross(cell[0],cell[1]);area=float(np.linalg.norm(normal));normal/=area
    h=pos@normal;h=h-float(np.mean(h))
    order=sorted(range(len(z)),key=lambda i:(h[i],int(z[i])))
    planes=[]
    for i in order:
        if not planes or h[i]-max(h[j] for j in planes[-1])>.15:planes.append([i])
        else:planes[-1].append(i)
    seq=tuple(tuple(sorted(int(z[i]) for i in plane)) for plane in planes)
    sign=1 if seq<=seq[::-1] else -1
    distances=np.asarray(g['edge_distance_A'],float);counts=np.asarray(g['neighbor_count'],float)
    a=float(np.linalg.norm(cell[0]));b=float(np.linalg.norm(cell[1]));gamma=math.degrees(math.acos(float(np.clip(cell[0]@cell[1]/(a*b),-1,1))))
    intrinsic={'lattice_a_A':a,'lattice_b_A':b,'lattice_gamma_deg':gamma,'area_per_atom_A2':area/len(z),
        'thickness_A':float(np.ptp(h)),'Z_mean':float(np.mean(z)),'Z_std':float(np.std(z)),
        'Z_min':float(np.min(z)),'Z_max':float(np.max(z)),'element_count':len(set(z.tolist())),
        'neighbor_count_mean':float(np.mean(counts)),'neighbor_count_std':float(np.std(counts)),
        'neighbor_distance_mean_A':float(np.mean(distances)) if len(distances) else None,
        'neighbor_distance_std_A':float(np.std(distances)) if len(distances) else None}
    return {'graph':g,'intrinsic':intrinsic,'planes':planes,'Z':z,'area':area,'canonical_sign':sign}

def contact_edges(m,side):
    """Source conversion or explicit user-supplied side edges; no SOC-side fallback."""
    reject_explicit_soc(m)
    side=normalized_contact_side(m,side)
    if 'noSOC_vbm_internal_reference_raw_eV' in m:
        low=required_finite(m.get('vacuum_low_eV'),'vacuum_low_eV');high=required_finite(m.get('vacuum_high_eV'),'vacuum_high_eV')
        vac={'low':low,'high':high,'equivalent_low_high':(low+high)/2}[side]
        return (required_finite(m.get('noSOC_vbm_internal_reference_raw_eV'),'noSOC internal VBM')-vac,
                required_finite(m.get('noSOC_cbm_internal_reference_raw_eV'),'noSOC internal CBM')-vac)
    if 'vacuum1_eV' in m and m.get('vacuum1_eV') is not None and 'vbm_raw' in m:
        low=required_finite(m.get('vacuum1_eV'),'vacuum1_eV');high=required_finite(m.get('vacuum2_eV'),'vacuum2_eV')
        vac={'low':low,'high':high,'equivalent_low_high':required_finite(m.get('vacuum_mean_eV'),'vacuum_mean_eV')}[side]
        return required_finite(m.get('vbm_raw'),'internal VBM')-vac,required_finite(m.get('cbm_raw'),'internal CBM')-vac
    supplied=m.get('contact_edges_eV',{})
    if side in supplied:return required_finite(supplied[side].get('vbm'),'contact VBM'),required_finite(supplied[side].get('cbm'),'contact CBM')
    if side=='equivalent_low_high':return required_finite(m.get('vbm_eV'),'VBM'),required_finite(m.get('cbm_eV'),'CBM')
    raise ValueError('Face-specific noSOC vacuum edges required for asymmetric input')

def monolayer_properties(m,side,position,intrinsic=None):
    reject_explicit_soc(m)
    side=normalized_contact_side(m,side)
    inf=intrinsic or graph_intrinsics(m);props=dict(inf['intrinsic'])
    gap=required_finite(m.get('gap_eV'),'monolayer gap');v=required_finite(m.get('vbm_eV'),'VBM');c=required_finite(m.get('cbm_eV'),'CBM')
    if abs((c-v)-gap)>1e-5:raise ValueError('Monolayer gap and vacuum edges disagree')
    vc,cc=contact_edges(m,side)
    if abs((cc-vc)-gap)>1e-5:raise ValueError('Contact side edges and monolayer gap disagree')
    eq=side=='equivalent_low_high'
    native_world_sign=1 if eq or (position=='lower' and side=='high') or (position=='upper' and side=='low') else -1
    canonical_code=0 if eq else (-1 if side=='low' else 1)*inf['canonical_sign']
    if m.get('vacuum_low_eV') is not None:
        step=required_finite(m.get('vacuum_high_eV'),'vacuum_high_eV')-required_finite(m.get('vacuum_low_eV'),'vacuum_low_eV')
    elif m.get('vacuum1_eV') is not None:step=required_finite(m.get('vacuum2_eV'),'vacuum2_eV')-required_finite(m.get('vacuum1_eV'),'vacuum1_eV')
    elif eq:step=0.0
    else:step=finite(m.get('vacuum_high_minus_low_eV'))
    dip=finite(m.get('dipole_z_eA'))
    props.update({'gap_eV':gap,'vbm_average_eV':v,'cbm_average_eV':c,'midgap_average_eV':(v+c)/2,
        'vbm_contact_eV':vc,'cbm_contact_eV':cc,'midgap_contact_eV':(vc+cc)/2,
        'electron_affinity_contact_eV':-cc,'ionization_energy_contact_eV':-vc,
        'vacuum_step_stack_eV':None if step is None else float(step)*native_world_sign,
        'dipole_density_stack_e_per_A':None if dip is None else dip*native_world_sign/inf['area'],
        'contact_side_canonical_code':canonical_code,
        'flip_from_chemical_canonical':0.0 if eq else float(inf['canonical_sign']*native_world_sign<0),
        'face_equivalent':float(m.get('face_equivalent',False))})
    planes=inf['planes'];z=inf['Z']
    contact_planes=planes if side=='low' else planes[::-1]
    props['contact_outer_Z_mean']=float(np.mean(z[contact_planes[0]]))
    props['contact_adjacent_Z_mean']=float(np.mean(z[contact_planes[min(1,len(contact_planes)-1)]]))
    composition={int(k):float(v/len(z)) for k,v in Counter(z.tolist()).items()}
    return props,composition,inf

def schema():
    columns={}
    def add(name,unit,group,meaning,default=None):columns[name]={'unit':unit,'group':group,'meaning':meaning,'default':default}
    for prop,(unit,group,meaning) in MONO_SPECS.items():
        for op in OPS:add(op+'__'+prop,unit,group,op+': '+meaning)
    for z in range(1,119):
        for op in OPS:add(op+'__atomic_fraction_Z'+str(z),'1','element_fraction',op+': fraction of element Z='+str(z),0.0)
    for name,(unit,group,meaning) in STACK_COLUMNS.items():add(name,unit,group,meaning)
    for pos in ('lower','upper'):
        for name in STRAIN_FIELDS:add(pos+'__initial_strain_'+name,'deg' if name=='angle_change_deg' else 'fraction','initial_strain',pos+' layer: '+name)
    add('reference_track_matched_strain','1','reference_track','1 matched-strain reference; 0 native/free')
    add('anderson_signed_eV','eV','physical_baseline','min contact CBM minus max contact VBM from inputs')
    add('anderson_nonnegative_eV','eV','physical_baseline','max(0,signed Anderson)')
    return {'version':'three-input-feature-schema-v1-2026-10-15','profile':'available_initial_stacking',
        'ordered_feature_names':list(columns),'columns':columns,'n_features':len(columns),
        'pair_operation':'signed_delta=upper-lower; null propagates to derived values',
        'physical_order':'lower/upper, independent of cosmetic A/B labels',
        'composition_default':'omitted atomic fractions are true zeros, not missing',
        'plane_cluster_tolerance_A':.15,'graph_parameters':'graphs/parameters.json',
        'excluded_input_groups':['dataset_identity','source_SOC_evidence','quality_bridge_flags','target_gap_or_sign',
                                 'source_alignment_type','bilayer_vh','tunneling_barrier','final_DFT_geometry',
                                 'auxiliary_mass_alpha_orbital_unaligned_SOC'],
        'profiles':list(PROFILES),'constant_schema_across_folds':True}

def pair_features(payload,intrinsics=None):
    reject_explicit_soc(payload)
    profile=payload.get('input_profile','available_initial_stacking')
    if profile not in PROFILES:raise ValueError('Unsupported input profile')
    st=payload['stacking'];monos=payload['monolayers'];lower=st['lower_layer']
    if lower not in ('A','B') or set(monos)!=set(('A','B')):raise ValueError('Exactly two monolayers and explicit lower_layer required')
    upper='B' if lower=='A' else 'A'
    descriptors={};compositions={};graphs={}
    for pos,label in [('lower',lower),('upper',upper)]:
        supplied=None if intrinsics is None else intrinsics[label]
        props,comp,inf=monolayer_properties(monos[label],st['contact_side_'+label],pos,supplied)
        descriptors[pos]=props;compositions[pos]=comp;graphs[pos]=inf['graph']
    values={}
    def pair(prop,a,b,omit_zero=False):
        vals={'lower':a,'upper':b,'mean':None if a is None or b is None else (a+b)/2,
              'signed_delta':None if a is None or b is None else b-a,
              'abs_delta':None if a is None or b is None else abs(b-a)}
        for op,value in vals.items():
            if not omit_zero or value:values[op+'__'+prop]=value
    for prop in MONO_SPECS:pair(prop,descriptors['lower'][prop],descriptors['upper'][prop])
    for z in range(1,119):pair('atomic_fraction_Z'+str(z),compositions['lower'].get(z,0.),compositions['upper'].get(z,0.),True)
    for key in STACK_COLUMNS:values[key]=None
    for pos in ('lower','upper'):
        for field in STRAIN_FIELDS:values[pos+'__initial_strain_'+field]=None
    if profile=='available_initial_stacking':
        allowed=st.get('continuous_stage') in ('qian_candidate_input','user_supplied_initial')
        if allowed:
            theta=finite(st.get('twist_angle_deg'));motif=finite(st.get('relative_motif_angle_mod120_deg'))
            if theta is not None:values.update({'twist_sin':math.sin(math.radians(theta)),'twist_cos':math.cos(math.radians(theta))})
            if motif is not None:values.update({'motif_sin3':math.sin(3*math.radians(motif)),'motif_cos3':math.cos(3*math.radians(motif))})
            uv=st.get('translation_uv')
            if uv is not None:
                if len(uv)!=2:raise ValueError('translation_uv has two fractional components')
                for axis,value in zip(('u','v'),uv):
                    angle=2*math.pi*(required_finite(value,'translation_'+axis)%1.)
                    values['translation_'+axis+'_sin']=math.sin(angle);values['translation_'+axis+'_cos']=math.cos(angle)
            d=finite(st.get('initial_contact_distance_A'))
            if d is not None and d<=0:raise ValueError('Initial interlayer distance must be positive')
            values['initial_contact_distance_A']=d
            for pos,label in [('lower',lower),('upper',upper)]:
                strain=st.get('strain_'+label) or {}
                for field in STRAIN_FIELDS:values[pos+'__initial_strain_'+field]=finite(strain.get(field))
    track=payload['reference_track']
    if track not in ('native_free','matched_strain'):raise ValueError('Reference track must be explicit')
    values['reference_track_matched_strain']=float(track=='matched_strain')
    L,U=descriptors['lower'],descriptors['upper']
    signed=min(L['cbm_contact_eV'],U['cbm_contact_eV'])-max(L['vbm_contact_eV'],U['vbm_contact_eV'])
    positive=L['gap_eV']>0 and U['gap_eV']>0
    values['anderson_signed_eV']=signed if positive else None
    values['anderson_nonnegative_eV']=max(0.,signed) if positive else None
    for name,value in values.items():
        if value is not None:required_finite(value,name)
    return {'values':values,'graphs':graphs,'profile':profile,'descriptor_lower':L,'descriptor_upper':U,
        'baseline_available':positive,'anderson_nonnegative_eV':max(0.,signed) if positive else None,
        'stacking_complete':all(values[k] is not None for k in ['translation_u_sin','translation_v_sin','initial_contact_distance_A']) and
                            (values['twist_sin'] is not None or values['motif_sin3'] is not None),
        'physical_order_labels':{'lower':lower,'upper':upper}}

def dense_values(values,feature_schema=None):
    spec=feature_schema or schema()
    return np.asarray([np.nan if (value:=values.get(name,spec['columns'][name]['default'])) is None else float(value)
                       for name in spec['ordered_feature_names']],dtype=float)

def swap_labels(payload):
    """Cosmetic A/B rename; all stacking quantities are physical upper-minus-lower."""
    swapped=json.loads(json.dumps(payload))
    swapped['monolayers']={'A':payload['monolayers']['B'],'B':payload['monolayers']['A']}
    st=swapped['stacking'];st['lower_layer']='B' if payload['stacking']['lower_layer']=='A' else 'A'
    for prefix in ('contact_side_','strain_'):
        a,b=payload['stacking'].get(prefix+'A'),payload['stacking'].get(prefix+'B')
        st[prefix+'A'],st[prefix+'B']=b,a
    return swapped
