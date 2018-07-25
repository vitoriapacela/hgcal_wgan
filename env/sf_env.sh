## Simons Foundation load environment
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

# Activate the virtual environment
source ~/tf9/bin/activate
