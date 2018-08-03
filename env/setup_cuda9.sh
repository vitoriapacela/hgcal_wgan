
## Simons Foundation GPU setup
## Author: Vitoria Barin Pacela
## vitoria.barinpacela@helsinki.fi
## 17.07.2018

#!/bin/bash 

# Load necessary modules

module load slurm
module load gcc
module load python3
module load cuda/9.0.176 
module load cudnn/v7.0-cuda-9.0

# Create a virtual environment in your home directory
virtualenv ~/tf9

# Activate the virtual environment
source ~/tf9/bin/activate

# Install dependencies into the environment
pip install tensorflow-gpu
pip install keras
pip install matplotlib
pip install gpustat
pip install setGPU
pip install ipykernel
