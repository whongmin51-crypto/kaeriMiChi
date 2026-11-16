"""Render the within-pair parity plot from the retained 40-pair source table."""
from pathlib import Path
import json,hashlib,sys
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(Path.home()/'.codex/skills/nature-figure/scripts'))
from audit_panel_alignment import require_matplotlib_panel_alignment

def plot_pair_response():
    source=ROOT/'analysis/pair_response.jsonl'
    rows=[json.loads(s) for s in source.read_text(encoding='utf-8').splitlines() if s.strip()]
    assert len(rows)==len({r['pair'] for r in rows})==40
    x=np.array([r['true_residual_range_eV'] for r in rows])
    y=np.array([r['predicted_residual_range_eV'] for r in rows])
    changes=np.array([r['applicability_changes'] for r in rows],dtype=bool)
    assert np.isfinite(x).all() and np.isfinite(y).all() and (x>=0).all() and (y>=0).all()
    assert changes.sum()==25
    mae=float(np.mean(np.abs(x-y)))
    limit=float(np.ceil(max(x.max(),y.max())/.2)*.2)
    grid=np.linspace(0,limit,401)
    mpl.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],
       'font.size':7,'axes.labelsize':7.5,'xtick.labelsize':7,'ytick.labelsize':7,
       'pdf.fonttype':42,'svg.fonttype':'none','axes.spines.top':True,'axes.spines.right':True,
       'axes.linewidth':.55,'legend.frameon':True})
    width_inches=89/25.4
    fig,ax=plt.subplots(figsize=(width_inches,width_inches))
    fig.subplots_adjust(left=.19,right=.96,bottom=.18,top=.95)
    ax.fill_between(grid,np.maximum(0,grid-mae),np.minimum(limit,grid+mae),
                    color='#D8D8D8',alpha=.9,linewidth=0,zorder=1)
    ax.plot([0,limit],[0,limit],color='#222222',ls='--',lw=.8,zorder=2)
    handles=[]
    for mask,color,label in [(~changes,'#7069FF','Class unchanged (15)'),(changes,'#FF565E','Class changes (25)')]:
        handles.append(ax.scatter(x[mask],y[mask],s=17,color=color,alpha=.8,
                                  edgecolors='white',linewidths=.2,label=label,zorder=3))
    handles.append(Patch(facecolor='#D8D8D8',edgecolor='none',label=f'±MAE ({mae:.3f} eV)'))
    legend=ax.legend(handles=handles,loc='upper left',bbox_to_anchor=(.018,.985),
                     fontsize=6.5,handletextpad=.6,labelspacing=.5,borderpad=.55,
                     borderaxespad=0,facecolor='white',edgecolor='#DDDDDD',framealpha=.96)
    legend.get_frame().set_linewidth(.45)
    ax.set(xlabel='DFT residual range (eV)',ylabel='Predicted residual range (eV)',
           xlim=(0,limit),ylim=(0,limit))
    ax.set_aspect('equal',adjustable='box')
    ticks=np.arange(0,limit+.01,.2)
    ax.set_xticks(ticks);ax.set_yticks(ticks)
    ax.tick_params(length=3,width=.55,color='#777777',pad=3)
    for spine in ax.spines.values():spine.set_color('#B0B0B0')
    ax.grid(False)
    out=ROOT/'figures'
    fig.canvas.draw()
    require_matplotlib_panel_alignment(fig,json_out=str(out/'pair_response.alignment.json'),strict=True)
    fig.savefig(out/'pair_response.pdf')
    fig.savefig(out/'pair_response.svg')
    fig.savefig(out/'pair_response.png',dpi=600)
    plt.close(fig)
    record={'n':40,'class_changes':25,'class_unchanged':15,'range_MAE_eV':mae,
            'shade':'Identity line plus/minus empirical mean absolute range error across the 40 pairs',
            'axis_limits':[0,limit],'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
            'all_coordinates_and_categories_preserved':True}
    (out/'pair_response.style.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
    print(json.dumps(record,indent=2))

if __name__=='__main__':plot_pair_response()
