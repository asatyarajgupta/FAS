# Cue-anchored validation of face presentation-attack detection

UCS532 Open Challenge, problem P14. Rolls: 1024030077, 1024030072.

**Question.** Face-PAD detectors generalise poorly to unseen data. Which image cues do they actually use? We train ResNet-18 and a Central Difference CNN (CDCN), measure the generalisation gap, remove cues from the same test images (blur, grayscale, face, background) and test Grad-CAM against those interventions.

## Data
CelebA-Spoof (Zhang et al., ECCV 2020), non-commercial research use only: https://github.com/ZhangYuanhan-AI/CelebA-Spoof (download link on that page).
Subset used (official train/test split, subject-disjoint, at most 40 images per subject, face crop from the supplied boxes, 128x128): train 13,199 images (340 subjects), val 2,382 (60), within-split test 9,803 (250 train-split subjects not used elsewhere), unseen test 8,329 (250 official-test subjects). Spoof-type annotations were not in our copy, so print/replay/mask attacks are pooled.
Replay-Attack/OULU-NPU were not available in time (data agreements), so "unseen" means the official test split (new people), not a new dataset.

## Run
Needs an NVIDIA GPU (tested on RTX 4060 8 GB), Python 3, `pip install -r requirements.txt`. About 2.5 hours in total.
```
bash run_all.sh /path/to/CelebA_Spoof      # folder containing Data/train and Data/test
```
Seeds are fixed and printed at the top of every log (`logs/`). Outputs go to `results/`. Arrays and checkpoints are written to `./cache` (not committed).
Quick check without the dataset or training: [DRIVE LINK to cache arrays and weights], set `PAD_CACHE` to that folder, then `python evaluate.py --models ... --damage ...` (see run_all.sh).

## Main results (frozen validation threshold; `results/gap_table.csv`)
| Model | Within-split ACER | Unseen ACER | Unseen ACER, oracle threshold | Gap ratio |
|---|---|---|---|---|
| ResNet-18 (ImageNet), 3 seeds | 2.6 +/- 0.6% | 20.4 +/- 2.0% | 15.4% | 5.9 / 8.4 / 10.7 |
| ResNet-18 from scratch (ablation) | 7.8% | 27.5% | 23.0% | 3.5 |
| CDCN, MSE + Sobel loss | 18.3% | 36.9% | 37.8% | 2.0 |
| CDCN, MSE only (ablation) | 1.2% | 14.4% | 5.2% | 12.2 |
CDCN: one seed only. Cue Reliance Index: `results/fig_cri.png`. Failure cases: `results/fig_failures_*.png`. Grad-CAM: `results/gradcam_*.csv`, `results/fig_gradcam.png`.

## Credits
- CDCN architecture adapted from https://github.com/ZitongYu/CDCN (`CVPR2020_paper_codes/models/CDCNs.py`, MIT, research use only). Differences: input 128x128, conv bias on, no final ReLU, our own loss and training.
- Idea of depth supervision: Liu et al., CVPR 2018. We do not implement their rPPG branch or registration layer.
- ResNet-18 and ImageNet weights: torchvision. Grad-CAM written from Selvaraju et al. (ICCV 2017), no external code.
- Dataset: CelebA-Spoof (above).

## Deviations and limitations
Not a cross-dataset test; train images are JPEG and test images PNG (a source-format confound we did not control); depth targets are all-ones/all-zeros (no 3D fitting); one seed for CDCN; no confidence intervals; interventions are blunt and the MSE-only CDCN is extremely sensitive to JPEG re-encoding and blur, so its CRI values are not clean single-cue measurements; Grad-CAM deletion for the MSE-only CDCN is inconclusive. Full list in the report.

## AI declaration
[Edit to match what is true.] An AI assistant (Claude) helped write the code in this repo and drafts of the report. The team ran every experiment and checked the outputs.

## Contributions
[Write what each person actually did.]

## Layout
`prepare_data.py` `train_resnet.py` `train_cdcn.py` `evaluate.py` `gradcam.py` `common.py` `cdcn_model.py` `run_all.sh` `notebooks/FAS.ipynb` (original exploratory notebook) `logs/` `results/`