#!/usr/bin/env python3

# Hunyuan 3D is licensed under the TENCENT HUNYUAN NON-COMMERCIAL LICENSE AGREEMENT
# except for the third-party components listed below.
# Hunyuan 3D does not impose any additional limitations beyond what is outlined
# in the repsective licenses of these third-party components.
# Users must comply with all terms and conditions of original licenses of these third-party
# components and must ensure that the usage of the third party components adheres to
# all relevant laws and regulations.

# For avoidance of doubts, Hunyuan 3D means the large language models and
# their software and algorithms, including trained model weights, parameters (including
# optimizer states), machine-learning model code, inference-enabling code, training-enabling code,
# fine-tuning enabling code and other elements of the foregoing made publicly available
# by Tencent in accordance with TENCENT HUNYUAN COMMUNITY LICENSE AGREEMENT.

import pytorch_lightning as pl
from torch.utils.data import Dataset, ConcatDataset, DataLoader
from torch.utils.data.distributed import DistributedSampler


class DataModuleFromConfig(pl.LightningDataModule):
    def __init__(
        self,
        batch_size=8,
        num_workers=4,
        train=None,
        validation=None,
        test=None,
        **kwargs,
    ):
        super().__init__()

        self.batch_size = batch_size
        self.num_workers = num_workers

        self.dataset_configs = dict()
        if train is not None:
            self.dataset_configs["train"] = train
        if validation is not None:
            self.dataset_configs["validation"] = validation
        if test is not None:
            self.dataset_configs["test"] = test

    def setup(self, stage):
        from src.utils.train_util import instantiate_from_config

        if stage in ["fit"]:
            dataset_dict = {}
            for k in self.dataset_configs:
                dataset_dict[k] = []
                for loader in self.dataset_configs[k]:
                    dataset_dict[k].append(instantiate_from_config(loader))
            self.datasets = dataset_dict
            print(self.datasets)
        else:
            raise NotImplementedError

    def train_dataloader(self):
        datasets = ConcatDataset(self.datasets["train"])
        sampler = DistributedSampler(datasets)
        return DataLoader(
            datasets,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            sampler=sampler,
            prefetch_factor=2,
            pin_memory=True,
        )

    def val_dataloader(self):
        datasets = ConcatDataset(self.datasets["validation"])
        sampler = DistributedSampler(datasets)
        return DataLoader(datasets, batch_size=4, num_workers=self.num_workers, shuffle=False, sampler=sampler)

    def test_dataloader(self):
        datasets = ConcatDataset(self.datasets["test"])
        return DataLoader(datasets, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)
