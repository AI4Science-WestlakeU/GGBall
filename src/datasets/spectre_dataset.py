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

class SpectreGraphDataset(InMemoryDataset):
    def __init__(self, dataset_name, split, root, transform=None, pre_transform=None, pre_filter=None):
        self.sbm_file = 'sbm_200.pt'
        self.planar_file = 'planar_64_200.pt'
        self.comm20_file = 'community_12_21_100.pt'
        self.dataset_name = dataset_name
        self.split = split
        self.num_graphs = 200
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        if self.dataset_name == 'tree':
            return ['tree.pkl']
        return ['train.pt', 'val.pt', 'test.pt']

    @property
    def processed_file_names(self):
            return [self.split + '.pt']

    def download(self):
        """
        Download raw qm9 files. Taken from PyG QM9 class
        """
        if self.dataset_name == 'sbm':
            raw_url = 'https://raw.githubusercontent.com/KarolisMart/SPECTRE/main/data/sbm_200.pt'
        elif self.dataset_name == 'planar':
            raw_url = 'https://raw.githubusercontent.com/KarolisMart/SPECTRE/main/data/planar_64_200.pt'
        elif self.dataset_name == 'comm20':
            raw_url = 'https://raw.githubusercontent.com/KarolisMart/SPECTRE/main/data/community_12_21_100.pt'
        elif self.dataset_name == 'tree':
            raw_url = 'https://raw.githubusercontent.com/AndreasBergmeister/graph-generation/main/data/tree.pkl'
            download_url(raw_url, self.raw_dir)
            return
        elif self.dataset_name == 'ego_small':
            raw_url = 'https://raw.githubusercontent.com/harryjo97/GDSS/master/data/ego_small.pkl'
        else:
            raise ValueError(f'Unknown dataset {self.dataset_name}')
        file_path = download_url(raw_url, self.raw_dir)

        if self.dataset_name == 'ego_small':
            with open(file_path, 'rb') as f:
                adjs = pickle.load(f)
                    
        else:
            adjs, eigvals, eigvecs, n_nodes, max_eigval, min_eigval, same_sample, n_max = torch.load(file_path)

        g_cpu = torch.Generator()
        g_cpu.manual_seed(0)

        test_len = int(round(self.num_graphs * 0.2))
        train_len = int(round((self.num_graphs - test_len) * 0.8))
        val_len = self.num_graphs - train_len - test_len
        indices = torch.randperm(self.num_graphs, generator=g_cpu)
        print(f'Dataset sizes: train {train_len}, val {val_len}, test {test_len}')
        train_indices = indices[:train_len]
        val_indices = indices[train_len:train_len + val_len]
        test_indices = indices[train_len + val_len:]

        train_data = []
        val_data = []
        test_data = []

        for i, adj in enumerate(adjs):
            if i in train_indices:
                train_data.append(adj)
            elif i in val_indices:
                val_data.append(adj)
            elif i in test_indices:
                test_data.append(adj)
            else:
                raise ValueError(f'Index {i} not in any split')

        torch.save(train_data, self.raw_paths[0])
        torch.save(val_data, self.raw_paths[1])
        torch.save(test_data, self.raw_paths[2])


    def process(self):
        if self.dataset_name == 'tree':
            with open(self.raw_paths[0], 'rb') as f:
                dataset = pickle.load(f)
            raw_dataset = dataset[self.split]
            data_list = []
            for graph in raw_dataset:
                # Extract the largest connected component
                if self.split=="train":
                    graph = graph.subgraph(max(nx.connected_components(graph), key=len))
                data = from_networkx(graph)
                # # Convert to PyG Data object
                # pdb.set_trace()
                n = data.num_nodes
                edge_index = data.edge_index
                y = torch.zeros([1, 0]).float()
                X = torch.ones(n, 1, dtype=torch.float)
                edge_attr = torch.zeros(edge_index.shape[-1], 2, dtype=torch.float)
                edge_attr[:, 1] = 1
                n_nodes = torch.tensor([n], dtype=torch.long)
                data = torch_geometric.data.Data(x=X, edge_index=edge_index, edge_attr=edge_attr,
                                                y=y, n_nodes=n)
                
                if self.pre_filter is not None and not self.pre_filter(data):
                    continue
                if self.pre_transform is not None:
                    data = self.pre_transform(data)

                data_list.append(data)

        elif self.dataset_name == 'ego_small':
            file_idx = {'train': 0, 'val': 1, 'test': 2}
            raw_dataset = torch.load(self.raw_paths[file_idx[self.split]])

            data_list = []
            for graph in raw_dataset:
                # Extract the largest connected component
                if self.split=="train":
                    graph = graph.subgraph(max(nx.connected_components(graph), key=len))
                data = from_networkx(graph)
                # # Convert to PyG Data object
                n = data.num_nodes
                edge_index = data.edge_index
                y = torch.zeros([1, 0]).float()
                X = torch.ones(n, 1, dtype=torch.float)
                edge_attr = torch.zeros(edge_index.shape[-1], 2, dtype=torch.float)
                edge_attr[:, 1] = 1
                n_nodes = torch.tensor([n], dtype=torch.long)
                data = torch_geometric.data.Data(x=X, edge_index=edge_index, edge_attr=edge_attr,
                                                y=y, n_nodes=n)
                
                if self.pre_filter is not None and not self.pre_filter(data):
                    continue
                if self.pre_transform is not None:
                    data = self.pre_transform(data)

                data_list.append(data)
                        
        else:
            file_idx = {'train': 0, 'val': 1, 'test': 2}
            raw_dataset = torch.load(self.raw_paths[file_idx[self.split]])

            data_list = []
            for adj in raw_dataset:
                n = adj.shape[-1]
                X = torch.ones(n, 1, dtype=torch.float)
                y = torch.zeros([1, 0]).float()
                edge_index, _ = torch_geometric.utils.dense_to_sparse(adj)
                edge_attr = torch.zeros(edge_index.shape[-1], 2, dtype=torch.float)
                edge_attr[:, 1] = 1
                num_nodes = n * torch.ones(1, dtype=torch.long)
                data = torch_geometric.data.Data(x=X, edge_index=edge_index, edge_attr=edge_attr,
                                                y=y, n_nodes=num_nodes)
                data_list.append(data)

                if self.pre_filter is not None and not self.pre_filter(data):
                    continue
                if self.pre_transform is not None:
                    data = self.pre_transform(data)

                data_list.append(data)
        torch.save(self.collate(data_list), self.processed_paths[0])



class SpectreGraphDataModule(AbstractDataModule):
    def __init__(self, cfg, n_graphs=200):
        self.cfg = cfg
        self.datadir = cfg.dataset.datadir
        self.debug = cfg.dataset.debug
        base_path = pathlib.Path(os.path.realpath(__file__)).parents[2]
        root_path = os.path.join(base_path, self.datadir)

        if self.debug:
            from torch.utils.data import Subset
            datasets = {'train': Subset(SpectreGraphDataset(dataset_name=self.cfg.dataset.name,
                                            split='train', root=root_path), list(range(1))),
                        'val': Subset(SpectreGraphDataset(dataset_name=self.cfg.dataset.name,
                                            split='val', root=root_path), list(range(1))),
                        'test': Subset(SpectreGraphDataset(dataset_name=self.cfg.dataset.name,
                                            split='test', root=root_path), list(range(1)))}
            
        else:
            datasets = {'train': SpectreGraphDataset(dataset_name=self.cfg.dataset.name,
                                            split='train', root=root_path),
                        'val': SpectreGraphDataset(dataset_name=self.cfg.dataset.name,
                                            split='val', root=root_path),
                        'test': SpectreGraphDataset(dataset_name=self.cfg.dataset.name,
                                            split='test', root=root_path)}
        # print(f'Dataset sizes: train {train_len}, val {val_len}, test {test_len}')

        super().__init__(cfg, datasets)
        self.inner = self.train_dataset

    def __getitem__(self, item):
        return self.inner[item]


class SpectreDatasetInfos(AbstractDatasetInfos):
    def __init__(self, datamodule, dataset_config):
        self.datamodule = datamodule
        self.name = 'nx_graphs'
        self.n_nodes = self.datamodule.node_counts()
        self.node_types = torch.tensor([1])               # There are no node types
        self.edge_types = self.datamodule.edge_counts()
        super().complete_infos(self.n_nodes, self.node_types)

