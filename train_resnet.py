import argparse
from torchvision.models import resnet18, ResNet18_Weights
import torch.nn as nn
from tqdm import tqdm
from common import *

ap = argparse.ArgumentParser()
ap.add_argument('--seed', type=int, default=0); ap.add_argument('--epochs', type=int, default=6)
ap.add_argument('--scratch', action='store_true', help='no ImageNet weights (ablation)')
a = ap.parse_args()
tag = f"resnet18{'_scratch' if a.scratch else ''}_seed{a.seed}"
start_log(f'train_{tag}'); set_seed(a.seed)
D = load_sets(); Xtr, ytr, Xva, yva = D['train']['X'], D['train']['y'], D['val']['X'], D['val']['y']

model = resnet18(weights=None if a.scratch else ResNet18_Weights.IMAGENET1K_V1)
model.fc = nn.Linear(model.fc.in_features, 1); model = model.to(device)
opt = torch.optim.Adam(model.parameters(), lr=1e-4)
lossf = nn.BCEWithLogitsLoss(pos_weight=torch.tensor((ytr == 0).sum() / (ytr == 1).sum(), device=device, dtype=torch.float32))
scaler = torch.amp.GradScaler('cuda'); best, best_state, bs = 1.0, None, 64

for ep in range(a.epochs):
    model.train(); perm = np.random.permutation(len(Xtr)); total = 0
    for i in tqdm(range(0, len(perm), bs), desc=f'epoch {ep+1}/{a.epochs}'):
        idx = np.sort(perm[i:i + bs]); x = to_batch(Xtr[idx])
        flip = torch.rand(len(idx), device=device) < 0.5; x[flip] = x[flip].flip(3)
        y = torch.from_numpy(ytr[idx]).float().to(device).unsqueeze(1)
        with torch.autocast('cuda', dtype=torch.float16): loss = lossf(model(x), y)
        opt.zero_grad(); scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        total += loss.item() * len(idx)
    s = predict_resnet(model, Xva); v = error_rates(s, yva, eer_threshold(s, yva))[2]
    print(f'epoch {ep+1}: train loss {total/len(Xtr):.4f} | val ACER {v:.4f}', flush=True)
    if v < best: best, best_state = v, {k: t.clone() for k, t in model.state_dict().items()}
model.load_state_dict(best_state)
torch.save(model.state_dict(), f'{CACHE}/{tag}.pt')
thr = eer_threshold(predict_resnet(model, Xva), yva)
summarize(tag, lambda X: predict_resnet(model, X), thr, D)