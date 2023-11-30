
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from models.gpt import VideoGPT, GPTLightningCLI
from models.data import VideoData

def main():

    val_checkpoint = ModelCheckpoint(
        dirpath='gpt_checkpoints',
        filename = "{epoch}-{step}-{val/loss:.6f}",
        every_n_epochs = 2,
        save_top_k = -1
    )

    best_checkpoint = ModelCheckpoint(
        dirpath='gpt_checkpoints',
        filename = "best_gpt_model",
        monitor = "val/loss",
        mode = "min",
        save_top_k = 1
    )

    cli = GPTLightningCLI(
        VideoGPT,
        VideoData,
        seed_everything_default = 123,
        run = False,
        trainer_defaults = {
            "max_epochs": 20,
            "max_steps": 2000,
            "accelerator": "auto",
            "devices": "auto", 
            "strategy": "auto",
            "callbacks":[val_checkpoint, best_checkpoint]
            }   
    )

    cli.trainer.fit(cli.model, datamodule=cli.datamodule)

if __name__ == '__main__':
    main()