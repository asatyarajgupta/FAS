import os
import numpy as np, pandas as pd, cv2, torch, torch.nn as nn, torch.nn.functional as F
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torchvision.models import resnet18
from common import *
from cdcn_model import CDCN

MODELS = ['resnet18_seed0', 'cdcn_seed0', 'cdcn_mse_seed0']
start_log('gradcam'); set_seed()
D = load_sets(); X, y, fb = D['test']['X'], D['test']['y'], D['test']['fb']
os.makedirs(RESULTS, exist_ok=True)

def build(tag, trained):
    res = tag.startswith('resnet')
    if res: m = resnet18(weights=None); m.fc = nn.Linear(m.fc.in_features, 1)
    else:   m = CDCN(theta=0.7)
    if trained: m.load_state_dict(torch.load(f'{CACHE}/{tag}.pt', map_location=device))
    m = m.to(device).eval()
    if res: return m, m.layer3, False, (lambda o: o.squeeze(1)), (lambda Z: predict_resnet(m, Z))
    return m, m.head, True, (lambda o: o.mean(dim=(1, 2, 3))), (lambda Z: predict_cdcn(m, Z))

def gradcam(model, target, use_input, score_fn, imgs, bs=8):
    store, cams = {}, []
    if use_input:
        def hook(m, inp): inp[0].retain_grad(); store['a'] = inp[0]
        h = target.register_forward_pre_hook(hook)
    else:
        def hook(m, inp, out): out.retain_grad(); store['a'] = out
        h = target.register_forward_hook(hook)
    for i in range(0, len(imgs), bs):
        model.zero_grad(); score_fn(model(to_batch(imgs[i:i + bs]))).sum().backward()
        A, G = store['a'], store['a'].grad
        cam = F.relu((G.mean(dim=(2, 3), keepdim=True) * A).sum(1, keepdim=True))
        cams.append(F.interpolate(cam, size=(128, 128), mode='bilinear', align_corners=False).detach().squeeze(1).cpu().numpy())
    h.remove(); return np.concatenate(cams)

rng = np.random.RandomState(0)
il = np.sort(rng.choice(np.where(y == 1)[0], 300, replace=False))
isp = np.sort(rng.choice(np.where(y == 0)[0], 300, replace=False))
XL, XS, FL, FS = np.array(X[il]), np.array(X[isp]), fb[il], fb[isp]

CAMS, SCORERS = {}, {}
for tag in MODELS:
    for trained in (True, False):
        torch.manual_seed(123)
        m, tgt, use_in, sf, scorer = build(tag, trained)
        name = tag if trained else tag + ' (random)'
        CAMS[name] = {'live': gradcam(m, tgt, use_in, sf, XL), 'spoof': gradcam(m, tgt, use_in, sf, XS)}
        if trained: SCORERS[tag] = scorer
        print(name, 'done', flush=True)

def masks(box, ring=0.25, n=128):
    x0, y0, x1, y1 = [int(v) for v in box]; w, h = x1 - x0, y1 - y0
    face = np.zeros((n, n), bool); face[y0:y1, x0:x1] = True
    ext = np.zeros((n, n), bool)
    ext[max(0, int(y0 - ring * h)):min(n, int(y1 + ring * h)), max(0, int(x0 - ring * w)):min(n, int(x1 + ring * w))] = True
    return face, ext & ~face, ~ext

def profile(cams, boxes):
    rows = []
    for c, b in zip(cams, boxes):
        if c.sum() <= 1e-8: continue
        c = c / c.sum(); f, mg, bg = masks(b)
        rows.append([c[f].sum(), c[mg].sum(), c[bg].sum(), f.mean(), mg.mean(), bg.mean()])
    return np.mean(rows, axis=0)

prof = []
for name, d in CAMS.items():
    for cls, boxes in (('live', FL), ('spoof', FS)):
        p = profile(d[cls], boxes)
        prof.append(dict(model=name, images=cls, face=p[0], margin=p[1], background=p[2],
                         face_area=p[3], margin_area=p[4], bg_area=p[5]))
prof = pd.DataFrame(prof); print(prof.round(3).to_string(index=False))
prof.to_csv(f'{RESULTS}/gradcam_profile.csv', index=False)

FRACS = [0.05, 0.10, 0.20, 0.30, 0.50]; rr = np.random.RandomState(1)
def deletion(scorer, order):
    base = scorer(XL); res = []
    for f in FRACS:
        k = int(f * 128 * 128); Xd = XL.copy()
        for i in range(len(Xd)):
            Xd[i].reshape(-1, 3)[np.argsort(-order[i].ravel())[:k]] = FILL
        res.append(float(np.mean(base - scorer(Xd)) / np.mean(base)))
    return res
noise = lambda c: c + 1e-9 * rr.rand(*c.shape)
rows = []
for tag in MODELS:
    for label, order in (('trained-model map', noise(CAMS[tag]['live'])),
                         ('random-weights map', noise(CAMS[tag + ' (random)']['live'])),
                         ('random pixels', rr.rand(len(XL), 128, 128))):
        v = deletion(SCORERS[tag], order)
        rows.append(dict(model=tag, deleted_by=label, **{f'{int(f*100)}%': round(x, 3) for f, x in zip(FRACS, v)},
                         mean=round(float(np.mean(v)), 3)))
dele = pd.DataFrame(rows); print(dele.to_string(index=False))
dele.to_csv(f'{RESULTS}/gradcam_deletion.csv', index=False)

def overlay(img, cam):
    c = (cam / (cam.max() + 1e-8) * 255).astype(np.uint8)
    return (0.55 * img + 0.45 * cv2.applyColorMap(c, cv2.COLORMAP_JET)[:, :, ::-1]).astype(np.uint8)
names = list(CAMS); n = 3; imgs = list(XL[:n]) + list(XS[:n])
fig, ax = plt.subplots(1 + len(names), 2 * n, figsize=(14, 2.4 * (1 + len(names))))
for j, im in enumerate(imgs):
    ax[0, j].imshow(im); ax[0, j].set_title('live' if j < n else 'spoof', fontsize=9)
    for r, nm in enumerate(names, start=1):
        ax[r, j].imshow(overlay(im, CAMS[nm]['live' if j < n else 'spoof'][j % n]))
        if j == 0: ax[r, j].set_ylabel(nm, fontsize=6)
for a in ax.flat: a.set_xticks([]); a.set_yticks([])
fig.savefig(f'{RESULTS}/fig_gradcam.png', dpi=200, bbox_inches='tight'); print('[done]')