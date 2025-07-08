import torch
import random
import numpy as np
import os.path as osp
import scipy.io as sio
import scipy.sparse as sp
from torch_geometric.utils import remove_self_loops, to_undirected
from torch_geometric.data import InMemoryDataset, download_url, Data


class LocalMatData(InMemoryDataset):
    r"""
        Use to load the ACM and UAI datasets appeared in AM-GCN
    Args:
        root (string): Root directory where the dataset should be saved.
        name (string): The name of the dataset (:obj:`"ACM"`, :obj:`"UAI"`).
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
    """

    url = 'https://github.com/shchur/gnn-benchmark/raw/master/data/npz/'

    def __init__(self, root, name, transform=None, pre_transform=None):
        assert name in ['ACM', 'UAI']
        self.name = name
        super(LocalMatData, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, self.name, 'raw')

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, self.name, 'processed')

    @property
    def raw_file_names(self):
        return '{}.mat'.format(self.name)

    @property
    def processed_file_names(self):
        return 'data.pt'

    def download(self):
        # download_url(self.url + self.raw_file_names, self.raw_dir)
        pass

    def process(self):
        data = read_local_mat_data(self.raw_dir, self.raw_file_names)
        data = data if self.pre_transform is None else self.pre_transform(data)
        data, slices = self.collate([data])
        torch.save((data, slices), self.processed_paths[0])

    def __repr__(self):
        return '{}()'.format(self.name)


def read_local_mat_data(folder, prefix):
    file_path = osp.join(folder,  '{}'.format(prefix.lower()))

    loader = sio.loadmat(file_path)
    x = torch.Tensor(loader['X']).to(torch.float)
    y = torch.Tensor(loader['Y'].squeeze()).long()
    adj = loader['adj']

    # Whether the loaded adj is symmetric
    assert np.sum((adj == np.transpose(adj)) == False) == 0

    # Process the edge_index
    coo_adj = sp.csr_matrix(adj).tocoo()
    edge_index = torch.tensor([coo_adj.row, coo_adj.col], dtype=torch.long)
    edge_index, _ = remove_self_loops(edge_index)
    edge_index = to_undirected(edge_index, x.size(0))  # Internal coalesce.

    # obtain the train data
    counter = []
    for i in range(len(set(list(loader['Y'].squeeze())))):
        counter.append(20)
    y_list = y.numpy()
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

    data = Data(x=x, edge_index=edge_index, y=y)
    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask

    return data


def sample_mask(index, num_nodes):
    mask = torch.zeros((num_nodes, ), dtype=torch.uint8)
    mask[index] = 1
    return mask