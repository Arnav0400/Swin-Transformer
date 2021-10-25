#!/bin/bash
#SBATCH --account=mbzuai
#SBATCH --partition=mbzuai
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --cpus-per-task 24
#SBATCH --job-name=swin
#SBATCH --time=20-00:00:00
#SBATCH --mem=150G
export PATH="/nfs/users/ext_arnav.chavan/miniconda3/bin:$PATH"
source activate
conda activate swin
python3 -m torch.distributed.launch --nproc_per_node=4 --master_port=$RANDOM --use_env prune.py --batch-size 256 --cfg configs/swin_tiny_patch4_window7_224.yaml --amp-opt-level O0 --data-path "/nfs/users/ext_arnav.chavan/imagenet" --output exps --tag "w1=2e-4_w2=4e-5"