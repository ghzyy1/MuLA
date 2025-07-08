import json
import warnings
from itertools import chain
from typing import Callable, List, Optional

import torch
import random
from torch_geometric.data import Data, InMemoryDataset, download_url
from torch_geometric.utils import to_undirected

class WikiCS(InMemoryDataset):
    r"""
    The semi-supervised Wikipedia-based dataset from the
    "Wiki-CS: A Wikipedia-Based Benchmark for Graph Neural Networks" paper,
    containing 11,701 nodes, 216,123 edges, 10 classes and 20 different training splits.
    
    Args:
        root (str): Root directory where the dataset should be saved.
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
        is_undirected (bool, optional): Whether the graph is undirected.
            (default: :obj:`True`)
        force_reload (bool, optional): Whether to re-process the dataset.
            (default: :obj:`False`)
    """

    url = 'https://github.com/pmernyei/wiki-cs-dataset/raw/master/dataset'

    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        is_undirected: Optional[bool] = True,
        force_reload: bool = False,
    ) -> None:
        if is_undirected is None:
            warnings.warn(
                f"The {self.__class__.__name__} dataset now returns an "
                f"undirected graph by default. Please explicitly specify "
                f"'is_undirected=False' to restore the old behavior.")
            is_undirected = True
        self.is_undirected = is_undirected
        super().__init__(root, transform, pre_transform,
                         force_reload=force_reload)
        self.load(self.processed_paths[0])

    @property
    def raw_file_names(self) -> List[str]:
        return ['data.json']

    @property
    def processed_file_names(self) -> str:
        return 'data_undirected.pt' if self.is_undirected else 'data.pt'

    # def download(self) -> None:
    #     for name in self.raw_file_names:
    #         download_url(f'{self.url}/{name}', self.raw_dir)

    def process(self) -> None:
        data = read_wiki_data(self.raw_paths[0], self.is_undirected)
        data = data if self.pre_transform is None else self.pre_transform(data)
        self.save([data], self.processed_paths[0])

    def __repr__(self):
        return f'{self.__class__.__name__}()'


def read_wiki_data(file_path, is_undirected):
    with open(file_path, 'r') as f:
        data = json.load(f)

    x = torch.tensor(data['features'], dtype=torch.float)
    y = torch.tensor(data['labels'], dtype=torch.long)

    edges = [[(i, j) for j in js] for i, js in enumerate(data['links'])]
    edges = list(chain(*edges))  # type: ignore
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    if is_undirected:
        edge_index = to_undirected(edge_index, num_nodes=x.size(0))

    # 获得训练数据
    class_counts = [20 for _ in range(len(set(y.tolist())))]
    train_list = []
    for idx, label in enumerate(y):
        if class_counts[int(label)] > 0:
            train_list.append(idx)
            class_counts[int(label)] -= 1
        if all(count == 0 for count in class_counts):
            break

    # 获得验证数据
    random_list = list(range(y.size(0)))
    curr_idx = 0
    val_list = []
    for idx in random_list:
        if idx not in train_list:
            val_list.append(idx)
            if len(val_list) == 500:
                break

    # 获得测试数据
    test_list = []
    for idx in random_list:
        if idx not in train_list + val_list:
            test_list.append(idx)
        if len(test_list) == 1000:
            break

    # 创建 train/val/test 掩码
    train_index = torch.LongTensor(train_list)
    val_index = torch.LongTensor(val_list)
    test_index = torch.LongTensor(test_list)
    train_mask = sample_mask(train_index, num_nodes=y.size(0))
    val_mask = sample_mask(val_index, num_nodes=y.size(0))
    test_mask = sample_mask(test_index, num_nodes=y.size(0))

    data = Data(x=x, edge_index=edge_index, y=y)
    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask

    return data


def sample_mask(index, num_nodes):
    mask = torch.zeros((num_nodes,), dtype=torch.bool)
    mask[torch.tensor(index)] = 1
    return mask
