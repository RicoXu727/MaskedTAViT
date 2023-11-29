
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.cli import LightningCLI
from pytorch_lightning.loggers import TensorBoardLogger

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

    logger = TensorBoardLogger("tb_logs", name="gpt_v0")

    cli = LightningClI(
        VideoGPT,
        VideoData,
        seed_everything_default = 123,
        run = False,
        trainer_defaults = {
            "max_epochs": 20,
            "accelerator": "auto",
            "devices": "auto", 
            "strategy": "auto",
            "callbacks":[val_checkpoint, best_checkpoint],
            "logger": logger
            }   
    )


if __name__ == '__main__':
    main()