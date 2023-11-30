
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.cli import LightningCLI

from models.vqvae import VQVAE
from models.data import VideoData

def main():

    val_checkpoint = ModelCheckpoint(
        dirpath='vqvae_checkpoints',
        filename = "{epoch}-{step}-{val/recon_loss:.6f}",
        every_n_epochs = 2,
        save_top_k = -1
    )

    best_checkpoint = ModelCheckpoint(
        dirpath='vqvae_checkpoints',
        filename = "best_vqvae_model",
        monitor = "val/recon_loss",
        mode = "min",
        save_top_k = 1
    )

    cli = LightningCLI(
        VQVAE, 
        VideoData,
        seed_everything_default = 123,
        run = False,
        trainer_defaults = {
            "max_epochs": 20,
            "accelerator": "auto",
            "devices": "auto", 
            "strategy": "auto",
            "callbacks":[val_checkpoint, best_checkpoint]
            }
    )

    cli.trainer.fit(cli.model, datamodule=cli.datamodule)

if __name__ == '__main__':
    main()