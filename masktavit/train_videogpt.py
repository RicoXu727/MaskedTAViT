
import argparse

from pytorch_lightning.callbacks import ModelCheckpoint

from models.gpt import VideoGPT, GPTLightningCLI
from models.data import VideoData

def main(args):

    ckpt_dirpath = args.ckpt_dirpath
    
    val_checkpoint = ModelCheckpoint(
        dirpath=ckpt_dirpath,
        filename = "{epoch}-{step}-{val/loss:.6f}",
        every_n_epochs = 1,
        save_top_k = -1
    )

    best_checkpoint = ModelCheckpoint(
        dirpath=ckpt_dirpath,
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
            "max_epochs": 30,
            "max_steps": 200000,
            "accelerator": "auto",
            "devices": "auto", 
            "strategy": "auto",
            "callbacks":[val_checkpoint, best_checkpoint]
            }   
    )

    cli.trainer.fit(cli.model, datamodule=cli.datamodule)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_dirpath', type=str, default='gpt_checkpoints', help='the path where the checkpoints to store')
    args = parser.parse_args()
    main(args)