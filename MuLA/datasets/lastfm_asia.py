import torch
import random
import numpy as np
import os.path as osp
from typing import Optional, Callable
from torch_geometric.data import InMemoryDataset, Data, download_url


class LastFMAsia(InMemoryDataset):
    r"""The LastFM Asia Network dataset introduced in the `"Characteristic
    Functions on Graphs: Birds of a Feather, from Statistical Descriptors to
    Parametric Models" <https://arxiv.org/abs/2005.07959>`_ paper.
    Nodes represent LastFM users from Asia and edges are friendships.
    It contains 7,624 nodes, 55,612 edges, 128 node features and 18 classes.

    Args:
        root (string): Root directory where the dataset should be saved.
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
    """

    url = 'https://graphmining.ai/datasets/ptg/lastfm_asia.npz'

    def __init__(self, root: str, transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None):
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, 'LastFMAsia', 'raw')

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, 'LastFMAsia', 'processed')

    @property
    def raw_file_names(self) -> str:
        return 'lastfm_asia.npz'

    @property
    def processed_file_names(self) -> str:
        return 'data.pt'

    def download(self):
        download_url(self.url, self.raw_dir)

    def process(self):
        data = np.load(self.raw_paths[0], 'r', allow_pickle=True)
        x = torch.from_numpy(data['features']).to(torch.float)
        y = torch.from_numpy(data['target']).to(torch.long)
        edge_index = torch.from_numpy(data['edges']).to(torch.long)
        edge_index = edge_index.t().contiguous()

        # obtain the train data
        counter = []
        y_list = y.numpy()
        for i in range(len(set(y_list))):
            counter.append(20)
        train_list = []
        for i in range(len(y_list)):
            if counter[y_list[i]] != 0:
                train_list.append(i)
                counter[y_list[i]] = counter[y_list[i]] - 1
            if max(counter) == 0:
                break
        # obtain the validation data
        random_list = random.sample(range(0, y.size(0)), y.size(0))
        curr_idx = 0
        val_list = []
        for i in range(len(random_list)):
            curr_idx = i
            if random_list[i] in train_list:
                continue
            else:
                val_list.append(random_list[i])
            if len(val_list) == 500:
                break
        # obtain the test data
        test_list = []
        for i in range(curr_idx + 1, len(random_list)):
            if random_list[i] in train_list:
                continue
            else:
                test_list.append(random_list[i])
            if len(test_list) == 1000:
                break

        # obtain the train/val/test mask
        train_index = torch.Tensor(train_list).long()
        val_index = torch.Tensor(val_list).long()
        test_index = torch.Tensor(test_list).long()
        train_mask = sample_mask(train_index, num_nodes=y.size(0))
        val_mask = sample_mask(val_index, num_nodes=y.size(0))
        test_mask = sample_mask(test_index, num_nodes=y.size(0))

        data = Data(x=x, y=y, edge_index=edge_index)
        data.train_mask = train_mask
        data.val_mask = val_mask
        data.test_mask = test_mask
        if self.pre_transform is not None:
            data = self.pre_transform(data)

        torch.save(self.collate([data]), self.processed_paths[0])


def sample_mask(index, num_nodes):
    mask = torch.zeros((num_nodes, ), dtype=torch.uint8)
    mask[index] = 1
    return mask