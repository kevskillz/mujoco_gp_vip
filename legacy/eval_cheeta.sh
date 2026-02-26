#!/bin/bash
#SBATCH --job-name=ppo_halfcheetah_eval
#SBATCH --output=logs/ppo_halfcheetah_eval_%j.out
#SBATCH --error=logs/ppo_halfcheetah_eval_%j.err

#SBATCH --partition=coe-gpu          # change if needed
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00

module purge
module load python/3.10              # change to your cluster's version

source /home/hice1/klobo8/scratch/mujoco_gp_vip/rl/bin/activate        # change to your venv path

mkdir -p logs

MODEL_PATH=${1:-models/halfcheetah_forward_ppo}
VECNORM_PATH=${2:-models/vecnorm_forward.pkl}
EPISODES=${3:-10}

echo "Starting evaluation at $(date)"
echo "Running on $(hostname)"

python eval.py \
    --model "$MODEL_PATH" \
    --vecnorm "$VECNORM_PATH" \
    --episodes "$EPISODES"

echo "Finished at $(date)"
