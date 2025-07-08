import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected
from scipy.sparse import csr_matrix
import os.path as osp
from typing import Callable, List, Optional, Union
import random
import numpy as np
import torch
import scipy.sparse as sp
from torch_geometric.utils import remove_self_loops, to_undirected
from torch_geometric.data import Data, InMemoryDataset, download_url
from torch_geometric.utils import coalesce


class WikipediaNetwork(InMemoryDataset):
    r"""The Wikipedia networks introduced in the
    `"Multi-scale Attributed Node Embedding"
    <https://arxiv.org/abs/1909.13021>`_ paper.
    Nodes represent web pages and edges represent hyperlinks between them.
    Node features represent several informative nouns in the Wikipedia pages.
    The task is to predict the average daily traffic of the web page.

    Args:
        root (str): Root directory where the dataset should be saved.
        name (str): The name of the dataset (:obj:`"chameleon"`,
            :obj:`"crocodile"`, :obj:`"squirrel"`).
        geom_gcn_preprocess (bool): If set to :obj:`True`, will load the
            pre-processed data as introduced in the `"Geom-GCN: Geometric
            Graph Convolutional Networks" <https://arxiv.org/abs/2002.05287>_`,
            in which the average monthly traffic of the web page is converted
            into five categories to predict.
            If set to :obj:`True`, the dataset :obj:`"crocodile"` is not
            available.
            If set to :obj:`True`, train/validation/test splits will be
            available as masks for multiple splits with shape
            :obj:`[num_nodes, num_splits]`. (default: :obj:`True`)
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
        force_reload (bool, optional): Whether to re-process the dataset.
            (default: :obj:`False`)

    """

    raw_url = 'https://graphmining.ai/datasets/ptg/wiki'
    processed_url = ('https://raw.githubusercontent.com/graphdml-uiuc-jlu/'
                     'geom-gcn/f1fc0d14b3b019c562737240d06ec83b07d16a8f')

    def __init__(
        self,
        root: str,
        name: str,
        geom_gcn_preprocess: bool = True,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        force_reload: bool = False,
    ) -> None:
        assert name.lower() in ['squirrel', 'crocodile']
        self.name = name.lower()
        super(WikipediaNetwork, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
    @property
    def raw_dir(self) -> str:
        if self.name == 'squirrel':
            return osp.join(self.root, 'squirrel', 'raw')
        elif self.name == 'crocodile':
            return osp.join(self.root, 'crocodile', 'raw')
        else:
            return osp.join(self.root, self.name, 'raw')
        
    @property
    def processed_dir(self) -> str:
       if self.name == 'squirrel':
            return osp.join(self.root, 'squirrel', 'processed')
       elif self.name == 'crocodile':
            return osp.join(self.root, 'crocodile', 'processed')
       else:
            return osp.join(self.root, self.name, 'processed')

    @property
    def raw_file_names(self) -> Union[List[str], str]:
        # if self.geom_gcn_preprocess:
        #     return (['out1_node_feature_label.txt', 'out1_graph_edges.txt'] +
        #             [f'{self.name}_split_0.6_0.2_{i}.npz' for i in range(10)])
        # else:
        return '{}.npz'.format(self.name)

    @property
    def processed_file_names(self) -> str:
        return 'data.pt'

    # def download(self) -> None:
    #     if self.geom_gcn_preprocess:
    #         for filename in self.raw_file_names[:2]:
    #             url = f'{self.processed_url}/new_data/{self.name}/{filename}'
    #             download_url(url, self.raw_dir)
    #         for filename in self.raw_file_names[2:]:
    #             url = f'{self.processed_url}/splits/{filename}'
    #             download_url(url, self.raw_dir)
    #     else:
    #         download_url(f'{self.raw_url}/{self.name}.npz', self.raw_dir)

    # def process(self):
    #     data = read_WikipediaNetwork_data(self.raw_dir, self.raw_file_names)
    #     data = data if self.pre_transform is None else self.pre_transform(data)
    #     data, slices = self.collate([data])
    #     torch.save((data, slices), self.processed_paths[0])

    def __repr__(self):
        return '{}()'.format(self.name)


      

    def process(self):
        print("Processing data...")
        # 加载原始数据
        data = np.load(self.raw_paths[0], 'r', allow_pickle=True)
        x = torch.from_numpy(data['features']).to(torch.float)
        y = torch.from_numpy(data['target']).to(torch.long)
        edge_index = torch.from_numpy(data['edges']).to(torch.long)
        edge_index = edge_index.t().contiguous()

        # 获取标签列表
        y_list = y.numpy()
        
        # 获取最大标签值
        max_label = int(max(y_list))

        # 获取训练集索引
        counter = []
        for i in range(max_label + 1):  # 确保长度覆盖所有标签值
            counter.append(20)
        train_list = []
        for i in range(len(y_list)):
            if counter[y_list[i]] != 0:
                train_list.append(i)
                counter[y_list[i]] -= 1
                
            if max(counter) == 0:
                break

        # 获取验证集索引
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

        # 获取测试集索引
        test_list = []
        for i in range(curr_idx + 1, len(random_list)):
            if random_list[i] in train_list:
                continue
            else:
                test_list.append(random_list[i])
                
            if len(test_list) == 1000:
                break

        # 获取 train/val/test 掩码
        train_index = torch.Tensor(train_list).long()
        val_index = torch.Tensor(val_list).long()
        test_index = torch.Tensor(test_list).long()
        train_mask = sample_mask(train_index, num_nodes=y.size(0))
        val_mask = sample_mask(val_index, num_nodes=y.size(0))
        test_mask = sample_mask(test_index, num_nodes=y.size(0))

        # 创建数据对象
        data = Data(x=x, y=y, edge_index=edge_index)
        data.train_mask = train_mask
        data.val_mask = val_mask
        data.test_mask = test_mask
        if self.pre_transform is not None:
            data = self.pre_transform(data)

        # 保存处理后的数据
        torch.save(self.collate([data]), self.processed_paths[0])
        

def sample_mask(index, num_nodes):
    mask = torch.zeros((num_nodes, ), dtype=torch.uint8)
    mask[index] = 1
    return mask