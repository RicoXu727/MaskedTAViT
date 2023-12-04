#!/bin/bash

current_env=$(conda info --envs | awk '$1=="*"{print $2}')

if [ "$current_env" != "base" ]; then
    echo "Activating venv..."
    source /home/shiwen/anaconda3/bin/activate venv
fi

# echo "Training the VQ-VAE..."
# rm -rf vqvae_checkpoints
# rm -rf lightning_logs
# python3 masktavit/train_vqvae.py

# echo "Training the VideoGPT..."
# rm -rf gpt_checkpoints
# rm -rf lightning_logs
# python3 masktavit/train_videogpt.py --model.n_cond_frames 2

echo "Training the MaskedVideoGPT without frame condition"
rm -rf mask_gpt_checkpoints
rm -rf lightning_logs
python3 masktavit/train_maskgpt.py 

