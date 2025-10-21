#!/bin/bash

# Source conda setup
source ~/miniconda3/etc/profile.d/conda.sh

# Activate environment
conda activate dreamt

# Run the experiment in background
nohup python new_experiments.py &

# Deactivate environment
conda deactivate
