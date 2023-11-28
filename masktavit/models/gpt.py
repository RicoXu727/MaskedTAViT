'''
Code referenced from https://github.com/wilson1yan/VideoGPT
'''
import os
import itertools
import numpy as np
from tqdm import tqdm
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler
import pytorch_lightning as pl

from .resnet import resnet34
from .attention import AttentionStack, LayerNorm, AddBroadcastPosEmbed
from .utils import shift_dim


class VideoGPT(pl.LightningModule):
    def __init__(
        self,
        resolution: int = 128,
        n_cond_frames: int = 10,
        hidden_dim: int = 576,
        heads: int = 4,
        layers: int = 8,
        dropout: float = 0.2,
        attn_type: str = 'full',
        attn_dropout: float = 0.3
    ):
        super().__init__()
        self.resolution = resolution
        self.n_cond_frames = n_cond_frames

        # Load VQ-VAE and set all parameters to no grad
        from .vqvae import VQVAE
        from .download import load_vqvae
        if not os.path.exists(args.vqvae):
            self.vqvae = load_vqvae(args.vqvae)
        else:
            self.vqvae =  VQVAE.load_from_checkpoint(args.vqvae)
        for p in self.vqvae.parameters():
            p.requires_grad = False
        self.vqvae.codebook._need_init = False
        self.vqvae.eval()

        # ResNet34 for frame conditioning
        self.use_frame_cond = n_cond_frames > 0
        if self.use_frame_cond:
            frame_cond_shape = (n_cond_frames,
                                resolution // 4,
                                resolution // 4,
                                240)
            self.resnet = resnet34(1, (1, 4, 4), resnet_dim=240)
            self.cond_pos_embd = AddBroadcastPosEmbed(
                shape=frame_cond_shape[:-1], embd_dim=frame_cond_shape[-1]
            )
        else:
            frame_cond_shape = None

        # VideoGPT transformer
        self.shape = self.vqvae.latent_shape

        self.fc_in = nn.Linear(self.vqvae.embedding_dim, hidden_dim, bias=False)
        self.fc_in.weight.data.normal_(std=0.02)

        self.attn_stack = AttentionStack(
            self.shape, hidden_dim, heads, layers, dropout,
            attn_type, attn_dropout, frame_cond_shape
        )

        self.norm = nn.LayerNorm(hidden_dim)

        self.fc_out = nn.Linear(hidden_dim, self.vqvae.n_codes, bias=False)
        self.fc_out.weight.data.copy_(torch.zeros(self.vqvae.n_codes, hidden_dim))

        # caches for faster decoding (if necessary)
        self.frame_cond_cache = None

        self.save_hyperparameters()

    def get_reconstruction(self, videos):
        return self.vqvae.decode(self.vqvae.encode(videos))

    def sample(self, n, batch=None):
        device = self.fc_in.weight.device

        cond = dict()
        if self.use_frame_cond:
            assert batch is not None
            video = batch['video']

            if self.use_frame_cond:
                cond['frame_cond'] = video[:, :, :self.n_cond_frames]

        samples = torch.zeros((n,) + self.shape).long().to(device)
        idxs = list(itertools.product(*[range(s) for s in self.shape]))

        with torch.no_grad():
            prev_idx = None
            for i, idx in enumerate(tqdm(idxs)):
                batch_idx_slice = (slice(None, None), *[slice(i, i + 1) for i in idx])
                batch_idx = (slice(None, None), *idx)
                embeddings = self.vqvae.codebook.dictionary_lookup(samples)

                if prev_idx is None:
                    # set arbitrary input values for the first token
                    # does not matter what value since it will be shifted anyways
                    embeddings_slice = embeddings[batch_idx_slice]
                    samples_slice = samples[batch_idx_slice]
                else:
                    embeddings_slice = embeddings[prev_idx]
                    samples_slice = samples[prev_idx]

                logits = self(embeddings_slice, samples_slice, cond,
                              decode_step=i, decode_idx=idx)[1]
                # squeeze all possible dim except batch dimension
                logits = logits.squeeze().unsqueeze(0) if logits.shape[0] == 1 else logits.squeeze()
                probs = F.softmax(logits, dim=-1)
                samples[batch_idx] = torch.multinomial(probs, 1).squeeze(-1)

                prev_idx = batch_idx_slice
            samples = self.vqvae.decode(samples)
            samples = torch.clamp(samples, -0.5, 0.5) + 0.5

        return samples # BCTHW in [0, 1]


    def forward(self, x, targets, cond, decode_step=None, decode_idx=None):
        if self.use_frame_cond:
            if decode_step is None:
                cond['frame_cond'] = self.cond_pos_embd(self.resnet(cond['frame_cond']))
            elif decode_step == 0:
                self.frame_cond_cache = self.cond_pos_embd(self.resnet(cond['frame_cond']))
                cond['frame_cond'] = self.frame_cond_cache
            else:
                cond['frame_cond'] = self.frame_cond_cache

        h = self.fc_in(x)
        h = self.attn_stack(h, cond, decode_step, decode_idx)
        h = self.norm(h, cond)
        logits = self.fc_out(h)

        loss = F.cross_entropy(shift_dim(logits, -1, 1), targets)

        return loss, logits

    def training_step(self, batch, batch_idx):
        self.vqvae.eval()
        x = batch['video']

        cond = dict()
        if self.use_frame_cond:
            cond['frame_cond'] = x[:, :, :self.n_cond_frames]

        with torch.no_grad():
            targets, x = self.vqvae.encode(x, include_embeddings=True)
            x = shift_dim(x, 1, -1)

        loss, _ = self(x, targets, cond)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.training_step(batch, batch_idx)
        self.log('val/loss', loss, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=3e-4, betas=(0.9, 0.999))
        assert hasattr(self.args, 'max_steps') and self.args.max_steps is not None, f"Must set max_steps argument"
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, self.args.max_steps)
        return [optimizer], [dict(scheduler=scheduler, interval='step', frequency=1)]

