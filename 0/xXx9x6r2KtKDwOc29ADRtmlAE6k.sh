#!/bin/bash
#SBATCH --job-name=llm_oper
#SBATCH -t 8:00:00
#SBATCH --gres=gpu:1
#SBATCH -C "A100-40GB|A100-80GB|H100|V100-16GB|V100-32GB|RTX6000|A40|L40S"
#SBATCH --mem-per-gpu 16G
#SBATCH -n 12
#SBATCH -N 1
echo "Launching AIsurBL"
hostname

export HF_HOME=/storage/ice-shared/vip-vvk/llm_storage/
export HF_TOKEN="${HF_TOKEN}"
export HUGGINGFACE_HUB_TOKEN="${HF_TOKEN}"

# Run Python script
C:\Python313\python.exe -m src.llm_mutation C:\Users\kevsk\Downloads\mujoco_gp_vip\sota/ExquisiteNetV2/network.py C:\Users\kevsk\Downloads\mujoco_gp_vip\sota/ExquisiteNetV2/models/network_xXx9x6r2KtKDwOc29ADRtmlAE6k.py 0\xXx9x6r2KtKDwOc29ADRtmlAE6k_model.txt --top_p 0.1 --temperature 0.17 --apply_quality_control False --inference_submission True
