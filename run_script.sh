#!/bin/bash

current_env=$(conda info --envs | awk '$1=="*"{print $2}')

if [ "$current_env" != "base" ]; then
    echo "Activating venv..."
    source /home/shiwen/anaconda3/bin/activate venv
fi

# rm -rf vqvae_checkpoints
# rm -rf lightning_logs

# echo "Training the VQ-VAE..."
# python3 masktavit/train_vqvae.py

echo "Training the VideoGPT..."
rm -rf gpt_checkpoints

python3 masktavit/train_videogpt.py --model.n_cond_frames 2

