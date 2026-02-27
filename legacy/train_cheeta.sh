#!/bin/bash
#SBATCH --job-name=ppo_halfcheetah
#SBATCH --output=logs/ppo_halfcheetah_%j.out
#SBATCH --error=logs/ppo_halfcheetah_%j.err

#SBATCH --partition=coe-gpu           # change if needed (e.g., compute, batch, etc.)
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8            # match n_envs if possible
#SBATCH --mem=32G
#SBATCH --time=8:00:00

#SBATCH --gres=gpu:1                 # remove if running CPU-only

module purge
module load python/3.10              # change to your cluster’s version
module load cuda/12.1                # only if using GPU
source /home/hice1/adantuluri7/scratch/mujoco_gp_vip/rl/bin/activate       # change to your venv path

mkdir -p logs

echo "Starting training at $(date)"
echo "Running on $(hostname)"

python train.py

echo "Finished at $(date)"
