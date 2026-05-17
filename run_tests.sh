#!/bin/bash
#SBATCH --export=/usr/local/cuda/bin
#SBATCH --gres=gpu:1

## FLASH ATTENTION ##

# Rebuild the project if build/release/flash_attention and build/release/flash_attention_datagen do not exist
if [ ! -f build/release/fa ] || [ ! -f build/release/fa_datagen ]; then
    cmake --build build 
fi

# Run datagen if data/fa does not exist
if [ ! -d data/fa ]; then
    ./build/release/fa_datagen -o data/fa -s 1024
fi

echo
echo "******************************"
echo "** Running tests for fa **"
echo "******************************"

# Loop through directories 0 to 8
for dir in $(seq 0 8)
do
    echo "Running test for directory $dir"
    ./build/release/fa_test -e data/fa/$dir/output.raw -i data/fa/$dir/Q.raw,data/fa/$dir/K.raw,data/fa/$dir/V.raw -t matrix
done
