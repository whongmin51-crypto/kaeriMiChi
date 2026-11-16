"""Plot accepted source-specific and source-macro errors with original intervals."""
from pathlib import Path
import csv
import hashlib
import json
import sys
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
TASK = ROOT.parent/'source_errors_redesign_2026-11-13'
sys.path.insert(0, str(Path.home()/'.codex/skills/nature-figure/scripts'))
from audit_panel_alignment import require_matplotlib_panel_alignment


def plot_source_errors():
    source = TASK/'source_errors_source.csv'
    if not source.exists():
        source = ROOT/'source_errors_source.csv'
    with source.open(encoding='utf-8-sig', newline='') as f:
        rows = list(csv.DictReader(f))
    cohorts = ['Janus', 'HetDB', 'Source macro']
    subtitles = ['500 records', '221 records', 'Equal source weighting']
    methods = ['Residual', 'Routed', 'Direct', 'Anderson']
    colors = ['#217A9D', '#C29B68', '#9AA3B6', '#68727B']
    assert len(rows) == 12 and len({(r['cohort'],r['method']) for r in rows}) == 12
    values = np.array([[float(r[k]) for k in ['MAE_eV','lower_95_eV','upper_95_eV']] for r in rows])
    assert np.isfinite(values).all() and (values[:,1] >= 0).all()
    assert (values[:,1] <= values[:,0]).all() and (values[:,0] <= values[:,2]).all()
    assert values[:,2].max() < .40
    mpl.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],
        'font.size':7.5,'axes.labelsize':8,'xtick.labelsize':7,'ytick.labelsize':7,
        'axes.linewidth':.6,'axes.spines.top':False,'axes.spines.right':False,
        'pdf.fonttype':42,'svg.fonttype':'none','legend.frameon':False})
    width_inches = 183/25.4
    fig,axs = plt.subplots(1,3,figsize=(width_inches,90/25.4),sharey=True)
    fig.subplots_adjust(left=.085,right=.985,bottom=.23,top=.78,wspace=.24)
    x = np.arange(4)
    plotted = []
    for i,(ax,cohort,subtitle) in enumerate(zip(axs,cohorts,subtitles)):
        rr = {r['method']:r for r in rows if r['cohort'] == cohort}
        assert set(rr) == set(methods)
        means = np.array([float(rr[m]['MAE_eV']) for m in methods])
        lo = np.array([float(rr[m]['lower_95_eV']) for m in methods])
        hi = np.array([float(rr[m]['upper_95_eV']) for m in methods])
        ax.bar(x,means,width=.64,color=colors,edgecolor='none',zorder=3)
        ax.errorbar(x,means,yerr=np.array([means-lo,hi-means]),fmt='none',
            ecolor='#303942',elinewidth=.85,capsize=2.2,capthick=.85,zorder=4)
        for j,method in enumerate(methods):
            ax.annotate(f'{means[j]:.3f}',(x[j],hi[j]),xytext=(0,5),
                textcoords='offset points',ha='center',va='bottom',fontsize=7.3,
                color='#205F79' if j == 0 else '#303942')
            plotted.append({'source':cohort,'pipeline':method,'MAE_eV':means[j],
                'lower_95':lo[j],'upper_95':hi[j]})
        ax.set_xticks(x,methods)
        ax.set_xlim(-.55,3.55)
        ax.set_ylim(0,.44)
        ax.set_yticks(np.arange(0,.41,.10))
        ax.tick_params(axis='x',length=0,pad=7,labelsize=6.6)
        ax.tick_params(axis='y',length=3,width=.6,color='#88929A',pad=3)
        ax.spines['left'].set_color('#AEB6BD')
        ax.spines['bottom'].set_color('#AEB6BD')
        ax.set_axisbelow(True)
        ax.grid(False)
        for level in [.10,.20,.30]:
            ax.axhline(level,color='#E5E9EC',lw=.5,zorder=0)
        ax.annotate(chr(97+i),(0,1),xycoords='axes fraction',xytext=(-21,23),
            textcoords='offset points',fontweight='bold',fontsize=10,va='bottom')
        ax.annotate(cohort,(.5,1),xycoords='axes fraction',xytext=(0,23),
            textcoords='offset points',ha='center',fontweight='bold',fontsize=9,va='bottom')
        ax.annotate(subtitle,(.5,1),xycoords='axes fraction',xytext=(0,10),
            textcoords='offset points',ha='center',fontsize=7.2,color='#64717C',va='bottom')
    axs[0].set_ylabel('Mean absolute gap error (eV)',labelpad=7)
    fig.text(.535,.075,'Lower MAE indicates better performance',ha='center',fontsize=7.2,color='#64717C')
    out = ROOT/'figures'
    out.mkdir(exist_ok=True)
    fig.canvas.draw()
    require_matplotlib_panel_alignment(fig,json_out=str(out/'source_errors.alignment.json'),
        require_panel_labels=True,strict=True)
    fig.savefig(out/'source_errors.pdf')
    fig.savefig(out/'source_errors.svg')
    fig.savefig(out/'source_errors.png',dpi=600)
    fig.savefig(out/'source_errors.tiff',dpi=600,pil_kwargs={'compression':'tiff_lzw'})
    plt.close(fig)
    # Retain the established machine-readable figure table at exact precision.
    (out/'source_errors.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in plotted),encoding='utf-8')
    record = {'cohorts':cohorts,'methods':methods,'estimates':12,'interval_bounds':24,
        'figure_size_mm':[183,90],'common_axis_eV':[0,.44],'display_labels':'MAE rounded to 3 decimals',
        'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
        'interval_definition':'Original 95% composition-group bootstrap percentile intervals, 1000 draws, seed 20261025',
        'all_values_and_intervals_preserved':True,'references':'Nature 2025 TabPFN Fig. 4a; Nature Communications 2023 14:3647 Fig. 2a; style-only'}
    (out/'source_errors.style.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
    print(json.dumps(record,indent=2))


if __name__=='__main__':
    plot_source_errors()
