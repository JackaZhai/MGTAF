from torch_geometric.data import Data


# 自定义Data类
class TripleData(Data):
    def __init__(self, edge_index1=None, x1=None, edge_attr1=None, edge_index2=None, x2=None, edge_attr2=None,
                 edge_index3=None, x3=None, edge_attr3=None, y=None):
        super(TripleData, self).__init__()
        # 三个图的邻接矩阵，节点特征，边特征，标签
        self.edge_index1 = edge_index1# 边的索引
        self.x1 = x1# 节点特征
        self.edge_attr1 = edge_attr1# 边的特征
        self.edge_index2 = edge_index2
        self.x2 = x2
        self.edge_attr2 = edge_attr2
        self.edge_index3 = edge_index3
        self.x3 = x3
        self.edge_attr3 = edge_attr3
        self.y = y # 标签

    # 自定义拼接步长
    def __inc__(self, key, value, *args, **kwargs):# 用于计算图的节点数
        if key == 'edge_index1':
            return self.x1.size(0)  # 节点数
        elif key == 'edge_index2':
            return self.x2.size(0)
        elif key == 'edge_index3':
            return self.x3.size(0)
        else:
            return super().__inc__(key, value)