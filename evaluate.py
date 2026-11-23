import argparse, os
import numpy as np, pandas as pd, torch, torch.nn as nn
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torchvision.models import resnet18
from common import *
from cdcn_model import CDCN

ap = argparse.ArgumentParser()
ap.add_argument('--models', nargs='+', default=['resnet18_seed0', 'cdcn_seed0'])
ap.add_argument('--damage', nargs='*', default=None, help='tags for the damage tests (default: all models)')
a = ap.parse_args()
dmg_tags = a.models if a.damage is None else a.damage
start_log('evaluate'); set_seed()
D = load_sets(); os.makedirs(RESULTS, exist_ok=True)
SETS = {'within-split': (D['within']['X'], D['within']['y'], D['within']['fb']),
        'unseen test':  (D['test']['X'],   D['test']['y'],   D['test']['fb'])}

def load_model(tag):
    if tag.startswith('resnet'):
        m = resnet18(weights=None); m.fc = nn.Linear(m.fc.in_features, 1)
        m.load_state_dict(torch.load(f'{CACHE}/{tag}.pt', map_location=device)); m = m.to(device).eval()
        return lambda X: predict_resnet(m, X)
    m = CDCN(theta=0.7); m.load_state_dict(torch.load(f'{CACHE}/{tag}.pt', map_location=device))
    m = m.to(device).eval()
    return lambda X: predict_cdcn(m, X)

def failure_figure(tag):
    s = np.load(f'{CACHE}/{tag}_scores_test.npy'); y, X = D['test']['y'], D['test']['X']
    fa = np.where(y == 0)[0][np.argsort(-s[y == 0])[:8]]     # attacks scored most "live"
    fr = np.where(y == 1)[0][np.argsort(s[y == 1])[:8]]      # live faces scored most "spoof"
    fig, ax = plt.subplots(2, 8, figsize=(16, 4.6))
    for r, (idx, nm) in enumerate(((fa, 'false accept'), (fr, 'false reject'))):
        for c, i in enumerate(idx):
            ax[r, c].imshow(X[i]); ax[r, c].axis('off'); ax[r, c].set_title(f'{nm}\nscore {s[i]:.2f}', fontsize=7)
    fig.suptitle(f'{tag}: most confident errors on the unseen test set')
    fig.savefig(f'{RESULTS}/fig_failures_{tag}.png', dpi=200, bbox_inches='tight'); plt.close(fig)

gap_rows, dmg_all = [], []
for tag in a.models:
    scorer = load_model(tag)
    thr = eer_threshold(scorer(D['val']['X']), D['val']['y'])
    r = summarize(tag, scorer, thr, D)
    w, u = r['within-split'], r['unseen test']
    gap_rows.append(dict(model=tag, within_ACER=w['ACER'], within_APCER=w['APCER'], within_BPCER=w['BPCER'],
        unseen_ACER=u['ACER'], unseen_APCER=u['APCER'], unseen_BPCER=u['BPCER'],
        gap_ratio=u['ACER'] / w['ACER'], unseen_ACER_oracle=r['oracle_ACER_unseen'],
        calibration_share=(u['ACER'] - r['oracle_ACER_unseen']) / (u['ACER'] - w['ACER'])))
    failure_figure(tag)
    if tag in dmg_tags:
        d = run_damage(scorer, thr, SETS); d['model'] = tag
        d.to_csv(f'{RESULTS}/damage_{tag}.csv', index=False); dmg_all.append(d)

gap = pd.DataFrame(gap_rows); gap.to_csv(f'{RESULTS}/gap_table.csv', index=False)
print(gap.round(4).to_string(index=False))

if dmg_all:
    dm = pd.concat(dmg_all); dm.to_csv(f'{RESULTS}/damage_all.csv', index=False)
    fig, axes = plt.subplots(1, 2, figsize=(14, 3.8), gridspec_kw={'wspace': 0.6})
    for ax, s in zip(axes, SETS):
        p = dm[(dm.set == s) & ~dm.intervention.isin(['ref unmodified', 'I0 re-encode'])]
        p = p.pivot(index='model', columns='intervention', values='CRI')
        im = ax.imshow(p.values, vmin=0, vmax=1, cmap='viridis')
        ax.set_xticks(range(p.shape[1])); ax.set_xticklabels(p.columns, rotation=30, ha='right', fontsize=8)
        ax.set_yticks(range(p.shape[0])); ax.set_yticklabels(p.index, fontsize=8); ax.set_title(f'CRI - {s}')
        for i in range(p.shape[0]):
            for j in range(p.shape[1]):
                ax.text(j, i, f'{p.values[i, j]:.2f}', ha='center', va='center', color='w', fontsize=9)
    fig.colorbar(im, ax=axes, shrink=0.8, pad=0.02, label='Cue Reliance Index')
    fig.savefig(f'{RESULTS}/fig_cri.png', dpi=200, bbox_inches='tight')
print('[done] results written to', RESULTS)