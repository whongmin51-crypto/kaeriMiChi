"""Render the retained independent-confirmation DFT contrasts."""
from pathlib import Path
import csv
import hashlib
import json
import sys
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parent
TASK = ROOT.parent / 'controlled_response_redesign_2026-11-12'
sys.path.insert(0, str(Path.home() / '.codex/skills/nature-figure/scripts'))
from audit_panel_alignment import require_matplotlib_panel_alignment


def plot_controlled_response():
    source = TASK / 'controlled_response_source.csv'
    if not source.exists():
        source = ROOT / 'controlled_response_source.csv'
    with source.open(encoding='utf-8-sig', newline='') as f:
        rows = list(csv.DictReader(f))
    pairs = ['P05', 'P06', 'P07', 'P08']
    arms = ['flip', 'slip', 'distance_minus', 'distance_plus', 'strain']
    labels = ['Flip', 'Slide', 'Near', 'Far', 'Strain']
    names = ['InTe / TeInAlSe', r'Ga$_2$SSe / In$_2$STe', r'In$_2$SeTe / SeInAlTe', r'In$_2$STe / SeInAlTe']
    fields = ['delta_gap_DFT_eV', 'delta_gA_eV', 'delta_r_mechanism_eV']
    colors = ['#616A73', '#C99754', '#287B9B']
    assert len(rows) == 20 and len({r['calculation_id'] for r in rows}) == 20
    assert all(r['numerical_level'] == 'Nref' and r['role'] == 'independent_confirmation' for r in rows)
    values = np.array([[float(r[f]) for f in fields] for r in rows])
    assert np.isfinite(values).all()
    assert np.allclose(values[:, 0], values[:, 1] + values[:, 2], atol=1e-12, rtol=0)
    mpl.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'DejaVu Sans'],
        'font.size': 7.5, 'axes.labelsize': 8, 'xtick.labelsize': 7, 'ytick.labelsize': 7.5,
        'pdf.fonttype': 42, 'svg.fonttype': 'none', 'axes.linewidth': .6,
        'axes.spines.top': False, 'axes.spines.right': False, 'legend.frameon': False})
    fig, axs = plt.subplots(2, 2, figsize=(183/25.4, 132/25.4), sharex=True)
    fig.subplots_adjust(left=.095, right=.975, bottom=.145, top=.825, wspace=.30, hspace=.45)
    for k, (ax, pair, name) in enumerate(zip(axs.ravel(), pairs, names)):
        rr = {r['arm']: r for r in rows if r['design_id'] == pair}
        assert set(rr) == set(arms)
        y = np.arange(5)
        for idx in [0, 2, 4]:
            ax.axhspan(idx-.48, idx+.48, color='#F4F6F7', lw=0, zorder=0)
        ax.axvline(0, color='#919AA1', lw=.7, zorder=1)
        for j, (field, color) in enumerate(zip(fields, colors)):
            yy = y + (j-1)*.235
            xx = np.array([float(rr[a][field])*1000 for a in arms])
            ax.barh(yy, xx, height=.19, color=color, edgecolor='none', zorder=3)
            if j == 2:
                for i, arm in enumerate(arms):
                    val = xx[i]
                    env = rr[arm]['epsilon_pair_sum_eV']
                    measured = bool(env)
                    if measured:
                        ax.errorbar(val, yy[i], xerr=float(env)*1000, fmt='none',
                            ecolor=color, elinewidth=.8, capsize=2, capthick=.8, zorder=5)
                    ax.plot(val, yy[i], marker='D', ms=3.2, mfc=color if measured else 'white',
                        mec=color, mew=.8, ls='none', zorder=6)
                    extent = float(env)*1000 if measured else 0
                    sign = 1 if val >= 0 else -1
                    ax.annotate(f'{val:+.1f}', (val + sign*extent, yy[i]),
                        xytext=(sign*4, 0), textcoords='offset points',
                        ha='left' if sign > 0 else 'right', va='center', color=color, fontsize=6.5)
        ax.set_yticks(y, labels)
        ax.set_ylim(4.58, -.65)
        ax.set_xlim(-330, 240)
        ax.set_xticks([-300, -200, -100, 0, 100, 200])
        ax.tick_params(axis='y', length=0, pad=5)
        ax.tick_params(axis='x', length=3, width=.6, color='#9DA5AB', labelbottom=True)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_color('#BBC1C6')
        ax.set_axisbelow(True)
        ax.grid(False)
        ax.annotate(chr(97+k), (0, 1), xycoords='axes fraction', xytext=(-28, 13),
            textcoords='offset points', fontsize=10, fontweight='bold', va='bottom')
        ax.annotate(pair, (0, 1), xycoords='axes fraction', xytext=(0, 13),
            textcoords='offset points', fontsize=9, fontweight='bold', va='bottom')
        ax.annotate(name, (1, 1), xycoords='axes fraction', xytext=(0, 13),
            textcoords='offset points', fontsize=8, ha='right', va='bottom', color='#424A51')
    fig.legend(handles=[Patch(facecolor=c, label=l) for c,l in zip(colors,
        ['DFT gap change', 'Anderson-reference change', 'Residual change'])],
        loc='upper center', bbox_to_anchor=(.535,.987), ncol=3, fontsize=8,
        handlelength=1.35, handleheight=.7, columnspacing=2.2, handletextpad=.65)
    fig.text(.535, .076, 'Change relative to the same-pair base (meV)', ha='center', fontsize=8)
    fig.legend(handles=[
        Line2D([],[],color=colors[2],marker='D',mfc=colors[2],ms=3.5,lw=.8,label='Measured numerical envelope'),
        Line2D([],[],color=colors[2],marker='D',mfc='white',ms=3.5,ls='none',label='Envelope unmeasured')],
        loc='lower center', bbox_to_anchor=(.535,.014), ncol=2, fontsize=7,
        handlelength=1.7, columnspacing=2.4, handletextpad=.7)
    out = ROOT/'figures'
    fig.canvas.draw()
    require_matplotlib_panel_alignment(fig, json_out=str(out/'controlled_response.alignment.json'),
        require_panel_labels=True, strict=True)
    fig.savefig(out/'controlled_response.pdf')
    fig.savefig(out/'controlled_response.svg')
    fig.savefig(out/'controlled_response.png', dpi=600)
    fig.savefig(out/'controlled_response.tiff', dpi=600, pil_kwargs={'compression':'tiff_lzw'})
    plt.close(fig)
    record = {'data_rows':20, 'plotted_components':60, 'panels':pairs, 'unit':'meV',
        'common_limits_meV':[-330,240], 'measured_envelopes':2, 'unmeasured_envelopes':18,
        'source_csv_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
        'decomposition_identity_pass':True, 'source_values_preserved':True,
        'residual_labels':'one decimal place, signed; exact values retained in source CSV',
        'reference_adaptation':'Style-only; original DFT data and numerical envelopes'}
    (out/'controlled_response.style.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
    print(json.dumps(record,indent=2))


if __name__ == '__main__':
    plot_controlled_response()
