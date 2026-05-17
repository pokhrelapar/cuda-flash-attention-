#!/bin/bash
#SBATCH --export=/usr/local/cuda/bin
#SBATCH --gres=gpu:1
## FLASH ATTENTION ## 
export TMPDIR=$HOME/tmp
mkdir -p $TMPDIR

ncu --set full -o fa_benchmark -f ./build/release/fa_benchmark 1024 1024
