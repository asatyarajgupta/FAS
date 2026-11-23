import os, sys, random, argparse
import numpy as np, cv2
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from common import SEED, IMG_SIZE, CACHE, set_seed

ap = argparse.ArgumentParser()
ap.add_argument('--data', required=True, help=r'folder that contains Data\train and Data\test')
args = ap.parse_args()
BASE = os.path.join(args.data, 'Data') if os.path.isdir(os.path.join(args.data, 'Data')) else args.data
EXT, MARGIN = ('.png', '.jpg', '.jpeg'), 0.3
set_seed(SEED); os.makedirs(CACHE, exist_ok=True)

def items_of(split, subj):
    out = []
    for cls, lab in (('live', 1), ('spoof', 0)):
        d = os.path.join(BASE, split, subj, cls)
        if os.path.isdir(d):
            out += [(os.path.join(d, f), lab, subj) for f in sorted(os.listdir(d)) if f.lower().endswith(EXT)]
    return out

def subset(split, subjects, cap=40):
    out = []
    for s in subjects:
        it = items_of(split, s); random.Random(SEED).shuffle(it); out += it[:cap]
    return out

def crop(item):
    path, lab, subj = item
    try:
        img = cv2.imread(path); H, W = img.shape[:2]
        x, y, w, h = [float(v) for v in open(path.rsplit('.', 1)[0] + '_BB.txt').read().split()[:4]]
        x, w, y, h = x * W / 224, w * W / 224, y * H / 224, h * H / 224
        x0, y0 = max(0, int(x - MARGIN * w)), max(0, int(y - MARGIN * h))
        x1, y1 = min(W, int(x + w + MARGIN * w)), min(H, int(y + h + MARGIN * h))
        if x1 - x0 < 10 or y1 - y0 < 10: return None
        c = cv2.resize(img[y0:y1, x0:x1], (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
        sx, sy = IMG_SIZE / (x1 - x0), IMG_SIZE / (y1 - y0)
        fb = [max(0, min(IMG_SIZE, v)) for v in (int((x - x0) * sx), int((y - y0) * sy),
                                                  int((x + w - x0) * sx), int((y + h - y0) * sy))]
        return c[:, :, ::-1], lab, fb, subj
    except Exception:
        return None

def save(items, name):
    with ThreadPoolExecutor(8) as ex: res = [r for r in tqdm(ex.map(crop, items), total=len(items), desc=name) if r]
    np.save(f'{CACHE}/{name}_X.npy', np.stack([r[0] for r in res]).astype(np.uint8))
    np.save(f'{CACHE}/{name}_y.npy', np.array([r[1] for r in res], dtype=np.int64))
    np.save(f'{CACHE}/{name}_fb.npy', np.array([r[2] for r in res], dtype=np.int64))
    np.save(f'{CACHE}/{name}_subj.npy', np.array([r[3] for r in res]))
    print(f'[data] {name}: {len(res)} images | live {sum(r[1] for r in res)} | subjects {len(set(r[3] for r in res))}'
          f' | skipped {len(items) - len(res)}', flush=True)

train_subs = sorted(os.listdir(os.path.join(BASE, 'train'))); random.Random(SEED).shuffle(train_subs)
test_subs = sorted(os.listdir(os.path.join(BASE, 'test')));   random.Random(SEED).shuffle(test_subs)
pool = train_subs[:400]
val_subs = set(sorted(pool)[:0])                     # placeholder replaced below
sub_sorted = sorted(pool); random.Random(SEED).shuffle(sub_sorted)
val_subs = set(sub_sorted[:int(0.15 * len(sub_sorted))])
tr_subs = [s for s in pool if s not in val_subs]; va_subs = [s for s in pool if s in val_subs]

assert not (set(tr_subs) & set(va_subs)) and not (set(pool) & set(test_subs[:250])) \
       and not (set(train_subs[400:650]) & set(pool)), 'subject overlap!'
print('[data] subject-disjointness checks passed')

save(subset('train', tr_subs), 'train')
save(subset('train', va_subs), 'val')
save(subset('test', test_subs[:250]), 'test')
save(subset('train', train_subs[400:650]), 'within')