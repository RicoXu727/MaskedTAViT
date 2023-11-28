#!/bin/bash

current_env=$(conda info --envs | awk '$1=="*"{print $2}')

if [ "$current_env" != "base" ]; then
    echo "Activating venv..."
    source /home/shiwen/anaconda3/bin/activate venv
fi

rm -rf lightning_logs

# tensorboard --logdir .

echo "Training the VQ-VAE..."
python3 masktavit/train_vqvae.py

echo "Training the VideoGPT..."


