import argparse

from tqdm import tqdm
import numpy as np

import torch
import torch.distributed as dist

from metric.fvd import get_fvd_logits, frechet_distance
from metric.pytorch_i3d import InceptionI3d

from models.gpt import VideoGPT
from models.data import VideoData
from models.utils import download

MAX_BATCH = 16

def main(args):
    '''
    n_trials: default 1, Number of trials to compute mean/std
    '''

    n_trials = args.n_trials
    ckpt = args.ckpt
    data_path = args.path

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.set_grad_enabled(False)

    #################### Load VideoGPT ########################################
    gpt = VideoGPT.load_from_checkpoint(ckpt).to(device)
    gpt.eval()
    
    loader = VideoData(data_path=data_path).test_dataloader()

    #################### Load I3D ########################################
    i3d = InceptionI3d(400, in_channels=3).to(device)
    i3d_pretrained_400_filepath = 'masktavit/models/i3d_pretrained_400.pt'
    i3d.load_state_dict(torch.load(i3d_pretrained_400_filepath, map_location=device))
    i3d.eval()

    #################### Compute FVD ###############################
    fvds = []

    pbar = tqdm(total=n_trials)
    for _ in range(n_trials):
        fvd = eval_fvd(i3d, gpt, loader, device)
        fvds.append(fvd)

        pbar.update(1)
        fvd_mean = np.mean(fvds)
        fvd_std = np.std(fvds)

        pbar.set_description(f"FVD {fvd_mean:.2f} +/- {fvd_std:.2f}")

    pbar.close()
    print(f"Final FVD {fvd_mean:.2f} +/- {fvd_std:.2f}")


def eval_fvd(i3d, videogpt, loader, device):

    batch = next(iter(loader))
    batch = {k: v.to(device) for k, v in batch.items()}

    fake_embeddings = []
    for i in range(0, batch['video'].shape[0], MAX_BATCH):
        fake = videogpt.sample(MAX_BATCH, {k: v[i:i+MAX_BATCH] for k, v in batch.items()})
        fake = fake.permute(0, 2, 3, 4, 1).cpu().numpy() # BCTHW -> BTHWC
        fake = (fake * 255).astype('uint8')
        fake_embeddings.append(get_fvd_logits(fake, i3d=i3d, device=device))
    fake_embeddings = torch.cat(fake_embeddings)

    real = batch['video'].to(device)
    real = real + 0.5
    real = real.permute(0, 2, 3, 4, 1).cpu().numpy() # BCTHW -> BTHWC
    real = (real * 255).astype('uint8')
    real_embeddings = get_fvd_logits(real, i3d=i3d, device=device)
    
    assert fake_embeddings.shape[0] == real_embeddings.shape[0] 

    fvd = frechet_distance(fake_embeddings.clone(), real_embeddings)
    return fvd.item()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_trials', default=3, help='')
    parser.add_argument('--path', default='/home/shiwen/MaskedTAViT/datasets/bair.hdf5', help='the path of hdf5 data')
    parser.add_argument('--ckpt', default='', help='the path for the checkpoint of gpt model')
    args = parser.parse_args()
    main(args)