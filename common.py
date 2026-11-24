import os, random, time
import numpy as np, pandas as pd, cv2, torch

SEED = 0
IMG_SIZE, FILL = 128, 127
CACHE = os.environ.get('PAD_CACHE', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cache'))
RESULTS = os.environ.get('PAD_RESULTS', 'results')
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def set_seed(seed=SEED):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    print(f'[seed] {seed} | device {device} | torch {torch.__version__}', flush=True)

def ld(name, key, mmap=True):
    return np.load(os.path.join(CACHE, f'{name}_{key}.npy'), mmap_mode='r' if mmap else None)

def load_sets():
    d = {n: dict(X=ld(n, 'X'), y=np.array(ld(n, 'y')), fb=np.array(ld(n, 'fb')))
         for n in ('train', 'val', 'test', 'within')}
    return d

MEAN = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

def to_batch(Xu8):
    x = torch.from_numpy(np.array(Xu8)).to(device).permute(0, 3, 1, 2).float() / 255
    return (x - MEAN) / STD

@torch.no_grad()
def predict_resnet(model, X, bs=128):
    model.eval()
    return torch.cat([torch.sigmoid(model(to_batch(X[i:i + bs]))).squeeze(1).cpu()
                      for i in range(0, len(X), bs)]).numpy()

@torch.no_grad()
def predict_cdcn(model, X, bs=32):
    model.eval()
    return torch.cat([model(to_batch(X[i:i + bs])).float().mean(dim=(1, 2, 3)).cpu()
                      for i in range(0, len(X), bs)]).numpy()

def error_rates(scores, y, thr):                 # live = positive
    live = scores >= thr
    apcer = float(np.mean(live[y == 0])); bpcer = float(np.mean(~live[y == 1]))
    return apcer, bpcer, (apcer + bpcer) / 2

def eer_threshold(scores, y):
    ths = np.quantile(scores, np.linspace(0, 1, 1001))
    gaps = [abs(error_rates(scores, y, t)[0] - error_rates(scores, y, t)[1]) for t in ths]
    return float(ths[int(np.argmin(gaps))])

# ---- interventions ----
def i0(X, fb):
    out = np.empty_like(X)
    for i in range(len(X)):
        _, e = cv2.imencode('.jpg', X[i][:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, 95])
        out[i] = cv2.imdecode(e, 1)[:, :, ::-1]
    return out
def i1(X, fb): return np.stack([cv2.GaussianBlur(x, (0, 0), 2.0) for x in X])
def i2(X, fb):
    g = np.stack([cv2.cvtColor(x, cv2.COLOR_RGB2GRAY) for x in X]); return np.repeat(g[..., None], 3, 3)
def i3(X, fb): return np.stack([cv2.blur(x, (7, 7)) for x in X])
def i4(X, fb):
    out = np.full_like(X, FILL)
    for k, (a, b, c, d) in enumerate(fb): out[k, b:d, a:c] = X[k, b:d, a:c]
    return out
def i5(X, fb):
    out = X.copy()
    for k, (a, b, c, d) in enumerate(fb): out[k, b:d, a:c] = FILL
    return out
INTERVENTIONS = {'ref unmodified': lambda X, fb: X, 'I0 re-encode': i0, 'I1 low-pass': i1,
                 'I2 grayscale': i2, 'I3 defocus': i3, 'I4 background removed': i4, 'I5 face removed': i5}

def run_damage(scorer, thr, sets, chunk=1000):
    rows = []
    for set_name, (X, y, fb) in sets.items():
        base = None
        for name, fn in INTERVENTIONS.items():
            t = time.time()
            s = np.concatenate([scorer(fn(np.array(X[i:i + chunk]), fb[i:i + chunk]))
                                for i in range(0, len(X), chunk)])
            a, b, acer = error_rates(s, y, thr)
            base = acer if base is None else base
            rows.append(dict(set=set_name, intervention=name, APCER=a, BPCER=b, ACER=acer,
                             CRI=(acer - base) / (0.5 - base)))
            print(f'[damage] {set_name} | {name}: ACER {acer:.4f} ({time.time()-t:.0f}s)', flush=True)
    return pd.DataFrame(rows)

import json, sys
class _Tee:
    def __init__(self, *s): self.s = s
    def write(self, d): [x.write(d) or x.flush() for x in self.s]
    def flush(self): [x.flush() for x in self.s]
def start_log(name):
    os.makedirs('logs', exist_ok=True)
    sys.stdout = _Tee(sys.__stdout__, open(f'logs/{name}.log', 'w', encoding='utf-8'))

def summarize(tag, scorer, thr, D):
    out = {'tag': tag, 'threshold': thr}
    for nm, key in (('within-split', 'within'), ('unseen test', 'test')):
        s = scorer(D[key]['X']); y = D[key]['y']
        a, b, acer = error_rates(s, y, thr)
        out[nm] = dict(APCER=a, BPCER=b, ACER=acer)
        np.save(f'{CACHE}/{tag}_scores_{key}.npy', s)
        if key == 'test':
            out['oracle_ACER_unseen'] = error_rates(s, y, eer_threshold(s, y))[2]
    os.makedirs(RESULTS, exist_ok=True)
    json.dump(out, open(f'{RESULTS}/{tag}_summary.json', 'w'), indent=2)
    print('[summary]', json.dumps(out), flush=True)
    return out