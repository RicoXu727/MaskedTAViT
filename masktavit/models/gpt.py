'''
Code referenced from https://github.com/wilson1yan/VideoGPT
Modified to fit for PyTorch Lightning 2.0
'''
import os
import itertools
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler
import pytorch_lightning as pl
from pytorch_lightning.cli import LightningCLI

from .resnet import resnet34
from .attention import AttentionStack, AddBroadcastPosEmbed
from .utils import shift_dim

from .vqvae import VQVAE

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
        attn_dropout: float = 0.3,
        vqvae_ckpt: str = "/home/shiwen/MaskedTAViT/vqvae_checkpoints/best_vqvae_model.ckpt",
        max_steps: int = -1
    ):
        super().__init__()
        self.resolution = resolution
        self.n_cond_frames = n_cond_frames

        # Load VQ-VAE and set all parameters to no grad
        self.vqvae = VQVAE.load_from_checkpoint(vqvae_ckpt)
        for para in self.vqvae.parameters():
            para.requires_grad = False
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

        self.max_steps = max_steps

        self.fc_out = nn.Linear(hidden_dim, self.vqvae.n_codes, bias=False)
        self.fc_out.weight.data.copy_(torch.zeros(self.vqvae.n_codes, hidden_dim))

        # caches for faster decoding (if necessary)
        self.frame_cond_cache = None

        self.save_hyperparameters()

    def get_reconstruction(self, videos):
        return self.vqvae.decode(self.vqvae.encode(videos))


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
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, self.max_steps)
        return [optimizer], [dict(scheduler=scheduler, interval='step', frequency=1)]

class GPTLightningCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.link_arguments("trainer.max_steps", "model.max_steps")
