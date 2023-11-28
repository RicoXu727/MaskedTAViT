'''
Code referenced from https://github.com/wilson1yan/VideoGPT
Modified to fit for PyTorch Lightning 2.0
'''

import argparse
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.cli import LightningCLI
from models.vqvae import VQVAE
from models.data import VideoData

def main():

    cli = LightningCLI(
        VQVAE, 
        VideoData,
        seed_everything_default = 123,
        run = False,
        trainer_defaults = {
            "max_epochs": 10,
            "accelerator": "auto",
            "devices": "auto", 
            "strategy": "auto"
            }
    )

    cli.trainer.fit(cli.model, datamodule=cli.datamodule)
    cli.trainer.test(ckpt_path="best", datamodule=cli.datamodule)

if __name__ == '__main__':
    main()