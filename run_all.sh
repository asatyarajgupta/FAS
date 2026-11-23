#!/usr/bin/env bash
# Usage: bash run_all.sh /path/to/CelebA_Spoof      (about 2 hours on one RTX 4060)
set -e
DATA=${1:?"usage: bash run_all.sh <folder containing Data/train and Data/test>"}
export PAD_CACHE=${PAD_CACHE:-./cache}
mkdir -p logs results
python prepare_data.py --data "$DATA"
for s in 0 1 2; do python train_resnet.py --seed $s; done
python train_resnet.py --seed 0 --scratch        # ablation 1: no ImageNet pretraining
python train_cdcn.py --seed 0                    # main CDCN
python train_cdcn.py --seed 0 --loss mse         # ablation 2: no Sobel gradient term
python evaluate.py --models resnet18_seed0 resnet18_seed1 resnet18_seed2 resnet18_scratch_seed0 cdcn_seed0 cdcn_mse_seed0 \
                   --damage resnet18_seed0 cdcn_seed0
# python gradcam.py        # added next