import os
import torch
import random
import shutil
import os.path as osp
import scipy.sparse as sp
from torch_sparse import SparseTensor
from typing import Optional, Callable, List
from torch_geometric.data import InMemoryDataset, Data, extract_zip


class AttributedGraphDataset(InMemoryDataset):
    r"""A variety of attributed graph datasets from the
    `"Scaling Attributed Network Embedding to Massive Graphs"
    <https://arxiv.org/abs/2009.00826>`_ paper.

    Args:
        root (string): Root directory where the dataset should be saved.
        name (string): The name of the dataset (:obj:`"Wiki"`, :obj:`"Cora"`
            :obj:`"CiteSeer"`, :obj:`"PubMed"`, :obj:`"BlogCatalog"`,
            :obj:`"PPI"`, :obj:`"Flickr"`, :obj:`"Facebook"`, :obj:`"Twitter"`,
            :obj:`"TWeibo"`, :obj:`"MAG"`).
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
    """

    datasets = {
        'wiki': '1EPhlbziZTQv19OsTrKrAJwsElbVPEbiV',
        'cora': '1FyVnpdsTT-lhkVPotUW8OVeuCi1vi3Ey',
        'citeseer': '1d3uQIpHiemWJPgLgTafi70RFYye7hoCp',
        'pubmed': '1DOK3FfslyJoGXUSCSrK5lzdyLfIwOz6k',
        'blogcatalog': '178PqGqh67RUYMMP6-SoRHDoIBh8ku5FS',
        'ppi': '1dvwRpPT4gGtOcNP_Q-G1TKl9NezYhtez',
        'flickr': '1tZp3EB20fAC27SYWwa-x66_8uGsuU62X',
        'facebook': '12aJWAGCM4IvdGI2fiydDNyWzViEOLZH8',
        'twitter': '1fUYggzZlDrt9JsLsSdRUHiEzQRW1kSA4',
        'tweibo': '1-2xHDPFCsuBuFdQN_7GLleWa8R_t50qU',
        'mag': '1ggraUMrQgdUyA3DjSRzzqMv0jFkU65V5',
    }

    def __init__(self, root: str, name: str,
                 transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None):
        self.name = name.lower()
        assert self.name in self.datasets.keys()
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, self.name.capitalize(), 'raw')

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, self.name.capitalize(), 'processed')

    @property
    def raw_file_names(self) -> List[str]:
        return ['attrs.npz', 'edgelist.txt', 'labels.txt']

    @property
    def processed_file_names(self) -> str:
        return 'data.pt'

    def download(self):
        from google_drive_downloader import GoogleDriveDownloader as gdd
        path = osp.join(self.raw_dir, f'{self.name}.zip')
        gdd.download_file_from_google_drive(self.datasets[self.name], path)
        extract_zip(path, self.raw_dir)
        os.unlink(path)
        path = osp.join(self.raw_dir, f'{self.name}.attr')
        if self.name == 'mag':
            path = osp.join(self.raw_dir, self.name)
        for name in self.raw_file_names:
            os.rename(osp.join(path, name), osp.join(self.raw_dir, name))
        shutil.rmtree(path)

    def process(self):
        import pandas as pd

        x = sp.load_npz(self.raw_paths[0])
        if x.shape[-1] > 10000 or self.name == 'mag':
            x = SparseTensor.from_scipy(x).to(torch.float)
        else:
            x = torch.from_numpy(x.todense()).to(torch.float)

        df = pd.read_csv(self.raw_paths[1], header=None, sep=None,
                         engine='python')
        edge_index = torch.from_numpy(df.values).t().contiguous()

        with open(self.raw_paths[2], 'r') as f:
            ys = f.read().split('\n')[:-1]
            ys = [[int(y) - 1 for y in row.split()[1:]] for row in ys]
            multilabel = max([len(y) for y in ys]) > 1

        if not multilabel:
            y = torch.tensor(ys).view(-1)
        else:
            num_classes = max([y for row in ys for y in row]) + 1
            y = torch.zeros((len(ys), num_classes), dtype=torch.float)
            for i, row in enumerate(ys):
                for j in row:
                    y[i, j] = 1.

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

        data = Data(x=x, edge_index=edge_index, y=y)
        data.train_mask = train_mask
        data.val_mask = val_mask
        data.test_mask = test_mask
        data = data if self.pre_transform is None else self.pre_transform(data)
        torch.save(self.collate([data]), self.processed_paths[0])

    def __repr__(self) -> str:
        return f'{self.name.capitalize()}()'


def sample_mask(index, num_nodes):
    mask = torch.zeros((num_nodes, ), dtype=torch.uint8)
    mask[index] = 1
    return mask