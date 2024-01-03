#!/bin/bash
#SBATCH --job-name=chromatin-accessibility-prediction
#SBATCH --output=logs/chromatin-accessibility-prediction
#SBATCH --error=logs/chromatin-accessibility-prediction
#SBATCH --mail-type=ALL
#SBATCH --account=my_account
#SBATCH --mail-user=my.user@my.email
#SBATCH --open-mode=append
#SBATCH --gpus=4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --partition=my_partition
#SBATCH --time=1-00:00:00
#SBATCH --mem-per-cpu=8G

. /usr/local/anaconda3/etc/profile.d/conda.sh

echo "sourced conda"

conda activate chromatin-accessibility-prediction

echo "activated conda environment"

cd {/path/to/your/repository/}

echo "moved in folder {/path/to/your/repository/}"

echo "executing: python main.py ...."

python main.py
