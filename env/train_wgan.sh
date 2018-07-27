#!/bin/bash
#SBATCH --job-name=wgan
#SBATCH -c 2
#SBATCH --gres=gpu:1 
#SBATCH -p gpu --exclude=workergpu[00-07] # use only V100
#SBATCH -t 7-0
#SBATCH --output train.out

#!/bin/bash

# Load necessary modules
module purge

module load slurm
module load gcc
module load python3
module load cuda/9.0.176 
module load cudnn/v7.0-cuda-9.0

# Activate the virtual environment
source ~/tf9/bin/activate
 
#Executable here
python ../wgan_hgcal.py

