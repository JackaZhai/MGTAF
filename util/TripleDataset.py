import torch
from torch_geometric.data import InMemoryDataset
from util.TripleData import TripleData 
import numpy as np
import scipy.io as sio
import scipy.sparse as sp

class TripleDataset(InMemoryDataset):

    def __init__(self, root, name1, name2, name3, delta1, delta2, delta3, k=None, transform=None, pre_transform=None):
        self.root = root
        self.delta1 = delta1
        self.name1 = name1
        self.delta2 = delta2
        self.name2 = name2
        self.delta3 = delta3
        self.name3 = name3
        if self.name2 == 'SR' or self.name1 == 'SR' or self.name3 == 'SR':
            self.k = k
        super().__init__(self.root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])


    # 保存文件名
    @property
    def processed_file_names(self):
        if self.name1 == 'SR':
            return [f'data(SR,{self.delta1},{self.k})_({self.name2},{self.delta2})_({self.name3},{self.delta3}).pt']
        elif self.name2 == 'SR':
            return [f'data({self.name1},{self.delta1})_(SR,{self.delta2},{self.k})_({self.name3},{self.delta3}).pt']
        elif self.name3 == 'SR':
            return [f'data({self.name1},{self.delta1})_({self.name2},{self.delta2})_(SR,{self.delta3},{self.k}).pt']


    def process(self):
        fMRI_h_file = r'data\fMRI_h.mat'
        fMRI_mat = sio.loadmat(fMRI_h_file)
        print("Loaded Data from", fMRI_h_file)
        """---------------------------"""
        brainNetSet1, lab = self.getBrainNetSet(self.name1)
        brainNetSet2, _ = self.getBrainNetSet(self.name2)
        brainNetSet3, _ = self.getBrainNetSet(self.name3)

        graph_num = fMRI_mat['lab'][0].shape[0]
        fMRI = fMRI_mat['fMRImciNc']
        data_list = []
        for i in range(graph_num):
            # 提取节点特征
            X = torch.tensor(fMRI[:, i][0].T, dtype=torch.float)  # 节点特征
            
            # 生成三种不同的边索引
            edge_index1, edge_index2, edge_index3 = (self.generate_edge_index(i, brainNetSet1, self.delta1),
                                                     self.generate_edge_index(i, brainNetSet2, self.delta2),
                                                     self.generate_edge_index(i, brainNetSet3, self.delta3))
            
            # 获取标签 (0 或 1，表示健康与否)
            Y = torch.tensor((lab > 0).astype('int'))[0, i]
            
            # 创建三联图数据结构
            triple_data = TripleData(edge_index1=edge_index1, x1=X, edge_index2=edge_index2,
                                     x2=X, edge_index3=edge_index3, x3=X, y=Y)
            data_list.append(triple_data)


        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    def getBrainNetSet(self, name):
        brainNetSet_file = rf'data\brainNetSet_{name}.mat'
        brainNetSet_mat = sio.loadmat(brainNetSet_file)
        print("Loaded Data from", brainNetSet_file)
        if name == 'SR':
            return brainNetSet_mat['brainNetSet'][:, self.k][0], brainNetSet_mat['lab']
        else:
            return brainNetSet_mat['brainNetSet'][:, 0][0], brainNetSet_mat['lab']

    def generate_edge_index(self, i, brainNetSet, threshold):
        A = brainNetSet[:, :, i]
        edge_matrix = sp.coo_matrix((A >= threshold).astype('int64'))
        return torch.tensor(np.vstack((edge_matrix.row, edge_matrix.col)), dtype=torch.int64)