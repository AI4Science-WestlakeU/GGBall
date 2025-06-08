import os
import pathlib

import torch
from torch.utils.data import random_split
import torch_geometric.utils
from torch_geometric.data import InMemoryDataset, download_url

from src.datasets.abstract_dataset import AbstractDataModule, AbstractDatasetInfos
from torch_geometric.utils import from_networkx
import pickle
import networkx as nx
import pdb
from src.models.hyperbolic_nn_plusplus.geoopt_plusplus.manifolds import PoincareBall




class HyperbolicDatasetPair(InMemoryDataset):
    manifold = PoincareBall()
    dim = 4

    def __init__(self, distance=0.6, std=0.7):
        self.distance = distance
        self.std = std

    def __len__(self):
        return 20000

    def __getitem__(self, idx):
        sign0 = (torch.rand(1) > 0.5).float() * 2 - 1
        sign1 = (torch.rand(1) > 0.5).float() * 2 - 1

        mean0 = torch.tensor([self.distance, self.distance, -self.distance, -self.distance]) * sign0
        mean1 = torch.tensor([-self.distance, -self.distance, self.distance, self.distance]) * sign1

        x0 = PoincareBall().wrapped_normal((20, 4), mean=mean0, std=self.std)
        x1 = PoincareBall().wrapped_normal((20, 4), mean=mean1, std=self.std)

        return {"x0": x0, "x1": x1}



class HyperbolicDataModule(AbstractDataModule):
    def __init__(self, cfg, n_graphs=200):
        self.cfg = cfg
        self.datadir = cfg.dataset.datadir
        datasets = HyperbolicDatasetPair()
        N = len(datasets)
        N_val = N_test = N // 10
        N_train = N - N_val - N_test

        data_seed = 42

        train_set, val_set, test_set = torch.utils.data.random_split(
            datasets,
            [N_train, N_val, N_test],
            generator=torch.Generator().manual_seed(data_seed),
        )

        base_path = pathlib.Path(os.path.realpath(__file__)).parents[2]
        root_path = os.path.join(base_path, self.datadir)


        datasets = {'train': train_set,
                    'val': val_set,
                    'test': test_set}
        # print(f'Dataset sizes: train {train_len}, val {val_len}, test {test_len}')

        super().__init__(cfg, datasets)
        self.inner = self.train_dataset

    def __getitem__(self, item):
        return self.inner[item]


class HyperbolicDatasetInfos(AbstractDatasetInfos):
    def __init__(self, datamodule, dataset_config):
        self.datamodule = datamodule
        super().complete_infos(torch.tensor([0.1, 0.1, 0.1,0.1, 0.1, 0.1,0.1, 0.1, 0.1,0.1]), torch.tensor([1]))

