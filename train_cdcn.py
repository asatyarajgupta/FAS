import argparse
from tqdm import tqdm
from common import *
from cdcn_model import CDCN, CDCNLoss

ap = argparse.ArgumentParser()
ap.add_argument('--seed', type=int, default=0); ap.add_argument('--epochs', type=int, default=8)
ap.add_argument('--loss', choices=['full', 'mse'], default='full', help='mse = ablation without Sobel term')
a = ap.parse_args()
tag = f"cdcn{'_mse' if a.loss == 'mse' else ''}_seed{a.seed}"
start_log(f'train_{tag}'); set_seed(a.seed)
D = load_sets(); Xtr, ytr, Xva, yva = D['train']['X'], D['train']['y'], D['val']['X'], D['val']['y']

model = CDCN(theta=0.7).to(device); lossf = CDCNLoss(use_grad=(a.loss == 'full')).to(device)
opt = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
scaler = torch.amp.GradScaler('cuda'); best, best_state, bs, accum = 1.0, None, 8, 4

def depth_targets(y):
    return torch.from_numpy(y).float().to(device).view(-1, 1, 1, 1).expand(-1, 1, 32, 32).contiguous()

for ep in range(a.epochs):
    model.train(); perm = np.random.permutation(len(Xtr)); total = 0; step = 0; opt.zero_grad()
    for i in tqdm(range(0, len(perm), bs), desc=f'epoch {ep+1}/{a.epochs}'):
        idx = np.sort(perm[i:i + bs]); x = to_batch(Xtr[idx])
        flip = torch.rand(len(idx), device=device) < 0.5; x[flip] = x[flip].flip(3)
        with torch.autocast('cuda', dtype=torch.float16): pred = model(x)
        loss = lossf(pred.float(), depth_targets(ytr[idx]))
        scaler.scale(loss / accum).backward(); step += 1
        if step % accum == 0: scaler.step(opt); scaler.update(); opt.zero_grad()
        total += loss.item() * len(idx)
    s = predict_cdcn(model, Xva); v = error_rates(s, yva, eer_threshold(s, yva))[2]
    print(f'epoch {ep+1}: train loss {total/len(Xtr):.4f} | val ACER {v:.4f}', flush=True)
    if v < best: best, best_state = v, {k: t.clone() for k, t in model.state_dict().items()}
    torch.cuda.empty_cache()
model.load_state_dict(best_state)
torch.save(model.state_dict(), f'{CACHE}/{tag}.pt')
thr = eer_threshold(predict_cdcn(model, Xva), yva)
summarize(tag, lambda X: predict_cdcn(model, X), thr, D)