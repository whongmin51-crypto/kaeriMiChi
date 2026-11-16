"""Reproduce the matched four-model comparison from fixed ensemble predictions."""
from pathlib import Path
import csv
import hashlib
import itertools
import json
import sys

import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[2]
REVISION = Path(__file__).resolve().parent
ROOT = REPO / 'research_plan/retrain_no_tony_2026-11-02'
OUT = ROOT / 'figures'
SKILL_SCRIPTS = Path.home() / '.codex/skills/nature-figure/scripts'
sys.path.insert(0, str(SKILL_SCRIPTS))
from audit_panel_alignment import require_matplotlib_panel_alignment
sys.path.insert(0, str(REPO / 'research_plan/step12_frozen_source_tests_2026-10-25'))
from statistics12 import GroupBootstrap, weights

MODELS = ['GANN', 'CGCNN', 'XGBoost', 'RF']
DISPLAY = {'GANN': 'GANN', 'CGCNN': 'CGCNN', 'XGBoost': 'XGBoost', 'RF': 'Random forest'}
COLORS = {'GANN': '#266F8A', 'CGCNN': '#6D6294', 'XGBoost': '#B88647', 'RF': '#6E777B'}
TASKS = [('residual_regression', 'Residual learning'), ('direct_gap_regression', 'Direct prediction')]

def read_rows(path):
    return [json.loads(x) for x in path.read_text(encoding='utf-8').splitlines() if x.strip()]

def save_json(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')

def save_csv(path, rows):
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

def input_sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def analyze():
    evaluated_path = ROOT / 'evaluation/sealed_final_test/evaluated_predictions.jsonl'
    evaluated = read_rows(evaluated_path)
    assert len(evaluated) == 742
    selected = [r for r in evaluated if r['frozen_prediction']['prediction_status'] == 'in_domain']
    assert len(selected) == 721
    selected_ids = {r['metadata']['heterostructure_id'] for r in selected}
    inputs = {evaluated_path.relative_to(REPO).as_posix(): input_sha(evaluated_path)}
    predictions = {}
    for model, (task, _) in itertools.product(MODELS, TASKS):
        path = ROOT / 'inference/sealed_final_test' / f'{model}_{task}_ensemble.jsonl'
        batch = read_rows(path)
        assert len(batch) == 742 and all(r['ensemble_seed_count'] == 5 for r in batch)
        predictions[model, task] = {r['heterostructure_id']: r['predicted_gap_eV'] for r in batch}
        assert len(predictions[model, task]) == 742
        inputs[path.relative_to(REPO).as_posix()] = input_sha(path)

    summary, contrasts, sample_rows = [], [], []
    for cohort, rr in [('supported_721', selected), ('admitted_742', evaluated)]:
        meta = [r['metadata'] for r in rr]
        ids = [r['heterostructure_id'] for r in meta]
        true = np.array([r['g_PBE_eV'] for r in meta])
        boot = GroupBootstrap(meta, replicates=1000)
        ww = weights(meta)
        errors = {}
        for model, (task, title) in itertools.product(MODELS, TASKS):
            pred = np.array([predictions[model, task][k] for k in ids])
            assert np.isfinite(pred).all() and (pred >= 0).all()
            err = np.abs(pred - true)
            errors[model, task] = err
            ci = boot.ci(boot.mean(err))['percentile_95']
            # Independent pair-then-source arithmetic check.
            source_maes = {}
            for source in ['Janus', 'HetDB']:
                pairs = sorted({r['material_pair_id'] for r in meta if r['dataset'] == source})
                source_maes[source] = float(np.mean([
                    np.mean([err[i] for i, r in enumerate(meta) if r['dataset'] == source and r['material_pair_id'] == pair])
                    for pair in pairs
                ]))
            mae = float(ww @ err)
            assert abs(mae - np.mean(list(source_maes.values()))) < 1e-12
            summary.append({'cohort': cohort, 'task': task, 'model': model, 'n': len(meta),
                            'MAE_eV': mae, 'lower_95': ci[0], 'upper_95': ci[1],
                            'Janus_MAE_eV': source_maes['Janus'], 'HetDB_MAE_eV': source_maes['HetDB']})
            if cohort == 'supported_721':
                for i, r in enumerate(meta):
                    sample_rows.append({'record_id': ids[i], 'source': r['dataset'],
                                        'material_pair_id': r['material_pair_id'],
                                        'composition_group': r['split_group_key'],
                                        'task': task, 'model': model, 'DFT_gap_eV': true[i],
                                        'predicted_gap_eV': pred[i], 'absolute_error_eV': err[i],
                                        'source_pair_weight': ww[i]})
        for task, _ in TASKS:
            for left, right in itertools.combinations(MODELS, 2):
                difference = errors[left, task] - errors[right, task]
                ci = boot.ci(boot.mean(difference))['percentile_95']
                contrasts.append({'cohort': cohort, 'task': task, 'model_A': left, 'model_B': right,
                                  'MAE_A_minus_B_eV': float(ww @ difference),
                                  'lower_95': ci[0], 'upper_95': ci[1]})

    expected = json.loads((ROOT / 'evaluation/sealed_final_test/main_metrics.json').read_text(encoding='utf-8'))
    original = expected['regression']['service']['metrics']['source_macro']['mae_eV']
    result = next(r for r in summary if r['cohort'] == 'supported_721' and r['model'] == 'GANN' and r['task'] == 'residual_regression')
    assert abs(result['MAE_eV'] - original) < 1e-12
    assert abs(result['MAE_eV'] - 0.08024548794133783) < 1e-12
    for r in selected:
        k = r['metadata']['heterostructure_id']
        assert abs(predictions['GANN', 'residual_regression'][k] - r['frozen_prediction']['predicted_PBE_gap_eV']) < 1e-12

    OUT.mkdir(exist_ok=True)
    save_csv(OUT / 'model_comparison.csv', summary)
    save_csv(OUT / 'model_comparison.paired_differences.csv', contrasts)
    save_csv(OUT / 'model_comparison.samples.csv', sample_rows)
    save_csv(OUT / 'model_comparison.cohort.csv', [
        {'record_id': r['metadata']['heterostructure_id'], 'source': r['metadata']['dataset'],
         'included': r['metadata']['heterostructure_id'] in selected_ids,
         'fixed_support_status': r['frozen_prediction']['prediction_status']}
        for r in evaluated])
    save_json(OUT / 'model_comparison.statistics.json', {
        'status': 'PASS', 'cohort_rule': 'previously fixed input-support rule, applied identically to all eight independent task ensembles',
        'admitted_n': 742, 'supported_n': 721, 'excluded_n': 21,
        'source_counts': {'Janus': 500, 'HetDB': 221},
        'composition_groups': len({r['metadata']['split_group_key'] for r in selected}),
        'physical_pairs': len({(r['metadata']['dataset'], r['metadata']['material_pair_id']) for r in selected}),
        'seed_ensemble_n': 5, 'bootstrap_replicates': 1000, 'bootstrap_seed': 20261025,
        'summary': summary, 'paired_differences': contrasts, 'input_hashes': inputs,
        'checks': {'independent_weighting_reproduced': True, 'GANN_service_reproduced': True,
                   'same_IDs_all_models': True, 'finite_predictions_all_models': True}
    })
    return [r for r in summary if r['cohort'] == 'supported_721']

def plot(summary):
    mpl.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'DejaVu Sans'],
                         'font.size': 8, 'axes.labelsize': 8, 'axes.titlesize': 9,
                         'xtick.labelsize': 7, 'ytick.labelsize': 8, 'legend.fontsize': 7,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.spines.left': False, 'axes.linewidth': .7,
                         'pdf.fonttype': 42, 'svg.fonttype': 'none', 'legend.frameon': False})
    width_inches, height_inches = 178 / 25.4, 88 / 25.4
    fig, axes = plt.subplots(1, 2, figsize=(width_inches, height_inches))
    fig.subplots_adjust(left=.145, right=.98, top=.80, bottom=.25, wspace=.52)
    for j, (ax, (task, title)) in enumerate(zip(axes, TASKS)):
        rr = {r['model']: r for r in summary if r['task'] == task}
        for i, model in enumerate(MODELS):
            r = rr[model]
            point = r['MAE_eV']
            ax.errorbar(point, i, xerr=[[point - r['lower_95']], [r['upper_95'] - point]],
                        fmt='o', ms=5.5, capsize=2.5, color=COLORS[model], elinewidth=1.2, zorder=3)
            ax.text(r['upper_95'] + .010, i, f'{point:.3f}', ha='left', va='center', fontsize=8, color=COLORS[model])
        ax.set_yticks(range(4), [DISPLAY[m] for m in MODELS])
        ax.set_ylim(3.55, -.55)
        ax.set_xlim(0, .28)
        ax.set_xticks([0, .05, .10, .15, .20, .25], ['0', '0.05', '0.10', '0.15', '0.20', '0.25'])
        ax.tick_params(axis='y', length=0, pad=8)
        ax.tick_params(axis='x', length=3, color='#62676C')
        ax.set_axisbelow(True)
        ax.grid(False)
        ax.set_xlabel('Source-macro MAE (eV)', labelpad=8)
        ax.set_title(title, loc='left', pad=17, fontweight='bold')
        ax.annotate(chr(97+j), xy=(0, 1), xycoords='axes fraction', xytext=(-45, 16),
                    textcoords='offset points', fontsize=9, fontweight='bold', ha='left', va='bottom')
    fig.text(.145, .91, 'Four-model comparison', fontsize=11, fontweight='bold', ha='left')
    fig.text(.98, .91, 'Lower MAE is better', fontsize=8, color='#505A61', ha='right')
    fig.text(.145, .095, 'Same 721 holdouts: 500 Janus + 221 HetDB  |  Five-seed ensembles', fontsize=7.5)
    fig.text(.145, .045, 'Bars: 95% composition-group bootstrap intervals', fontsize=7.5, color='#505A61')
    fig.canvas.draw()
    require_matplotlib_panel_alignment(fig, json_out=str(OUT / 'model_comparison.alignment.json'), strict=True)
    fig.savefig(OUT / 'model_comparison.pdf')
    fig.savefig(OUT / 'model_comparison.svg')
    fig.savefig(OUT / 'model_comparison.png', dpi=600)
    plt.close(fig)

if __name__ == '__main__':
    result = analyze()
    plot(result)
    print(json.dumps(result, indent=2))
