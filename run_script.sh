#!/bin/bash

current_env=$(conda info --envs | awk '$1=="*"{print $2}')

if [ "$current_env" != "base" ]; then
    echo "Activating venv..."
    source /home/shiwen/anaconda3/bin/activate venv
fi

rm -rf lightning_logs

# echo "Training the VQ-VAE..."
# rm -rf vqvae_checkpoints
# python3 masktavit/train_vqvae.py

# echo "Training the VideoGPT without the frame condition..."
# rm -rf gpt_checkpoints
# python3 masktavit/train_videogpt.py \
#         --ckpt_dirpath 'gpt_checkpoints' 

# echo "Training the VideoGPT with the frame condition..."
# rm -rf gpt_frame_checkpoints
# python3 masktavit/train_videogpt.py \
#         --ckpt_dirpath 'gpt_frame_checkpoints' \
#         --model.n_cond_frames 2 

echo "Training the VideoGPT with the distance mask..."
rm -rf gpt_mask_checkpoints
python3 masktavit/train_videogpt.py \
        --ckpt_dirpath 'gpt_mask_checkpoints' \
        --model.dist_mask True

# echo "Training the VideoGPT with the frame condition and distance mask..."
# rm -rf gpt_mask_frame_checkpoints
# python3 masktavit/train_videogpt.py \
#         --ckpt_dirpath 'gpt_mask_frame_checkpoints' \
#         --model.n_cond_frames 2 
#         --model.dist_mask True


