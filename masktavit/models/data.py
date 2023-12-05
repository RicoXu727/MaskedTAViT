'''
Code referenced from https://github.com/wilson1yan/VideoGPT
Modified to fit for PyTorch Lightning 2.0
'''

import h5py
import numpy as np

import torch
import torch.utils.data as data
import torch.distributed as dist
import pytorch_lightning as pl


class HDF5Dataset(data.Dataset):
    """ Generic dataset for data stored in h5py as uint8 numpy arrays.
    Reads videos in {0, ..., 255} and returns in range [-0.5, 0.5] """
    def __init__(self, data_file, sequence_length, train=True, resolution=64):
        """
        Args:
            data_file: path to the pickled data file with the
                following format:
                {
                    'train_data': [B, H, W, 3] np.uint8,
                    'train_idx': [B], np.int64 (start indexes for each video)
                    'test_data': [B', H, W, 3] np.uint8,
                    'test_idx': [B'], np.int64
                }
            sequence_length: length of extracted video sequences
        """
        super().__init__()
        self.train = train
        self.sequence_length = sequence_length
        self.resolution = resolution

        # read in data
        self.data_file = data_file
        self.data = h5py.File(data_file, 'r')
        self.prefix = 'train' if train else 'test'
        self._images = self.data[f'{self.prefix}_data']
        self._idx = self.data[f'{self.prefix}_idx']
        self.size = len(self._idx)

    @property
    def n_classes(self):
        raise Exception('class conditioning not support for HDF5Dataset')

    def __getstate__(self):
        state = self.__dict__
        state['data'] = None
        state['_images'] = None
        state['_idx'] = None
        return state

    def __setstate__(self, state):
        self.__dict__ = state
        self.data = h5py.File(self.data_file, 'r')
        self._images = self.data[f'{self.prefix}_data']
        self._idx = self.data[f'{self.prefix}_idx']

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        start = self._idx[idx]
        end = self._idx[idx + 1] if idx < len(self._idx) - 1 else len(self._images)
        assert end - start >= 0

        start = start + np.random.randint(low=0, high=end - start - self.sequence_length)
        assert start < start + self.sequence_length <= end
        video = torch.tensor(self._images[start:start + self.sequence_length])
        return dict(video=preprocess(video, self.resolution))



class VideoData(pl.LightningDataModule):

    def __init__(
        self,
        data_path: str = '/home/shiwen/MaskedTAViT/datasets/bair.hdf5',
        sequence_length: int = 16,
        resolution: int = 64,
        batch_size: int = 32,
        num_workers: int = 8,
    ):
        super().__init__()
        self.data_path = data_path
        self.sequence_length = sequence_length
        self.resolution = resolution
        self.batch_size = batch_size
        self.num_workers = num_workers
        
    def _dataset(self, train):
        dataset = HDF5Dataset(
            self.data_path,
            self.sequence_length,
            train = train,
            resolution = self.resolution
        )
        return dataset

    def _dataloader(self, train):
        dataset = self._dataset(train)
        if dist.is_initialized():
            sampler = data.distributed.DistributedSampler(
                dataset, num_replicas=dist.get_world_size(), rank=dist.get_rank()
            )
        else:
            sampler = None
        dataloader = data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers = self.num_workers,
            pin_memory=True,
            sampler=sampler,
            shuffle=sampler is None
        )
        return dataloader
    
    def train_dataloader(self):
        return self._dataloader(True)

    def val_dataloader(self):
        return self._dataloader(False)

    def test_dataloader(self):
        return self.val_dataloader()

