import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class GCN(nn.Module):
    """图卷积网络层
    
    使用GCN进行特征提取，包含归一化和非线性激活。
    
    参数:
        in_channels (int): 输入特征维度
        out_channels (int): 输出特征维度
    """
    def __init__(self, in_channels, out_channels):
        super(GCN, self).__init__()
        self.conv = GCNConv(in_channels, out_channels)
        self.bn = nn.BatchNorm1d(out_channels)
        
    def forward(self, x, edge_index):
        """前向传播
        
        参数:
            x (Tensor): 节点特征矩阵
            edge_index (Tensor): 边索引
            
        返回:
            Tensor: 处理后的节点特征
        """
        x = self.conv(x, edge_index)
        x = self.bn(x)
        x = F.relu(x)
        return x
    
class GraphTransformer(nn.Module):
    """图Transformer模型
    
    通过多层自注意力机制处理图结构数据，捕获节点间的长距离依赖关系。
    
    参数:
        in_channels (int): 输入特征维度
        hidden_channels (int): 隐藏层维度，默认为32
        num_heads (int): 注意力头数量，默认为4
        dropout (float): Dropout比率，默认为0.1
        layers (int): Transformer层数，默认为1
    """
    def __init__(self, in_channels, hidden_channels=32, num_heads=4, dropout=0.1, layers=1):
        super(GraphTransformer, self).__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.num_heads = num_heads
        self.head_dim = hidden_channels // num_heads
        self.layers = layers
        
        # 输入特征的线性投影层
        self.input_proj = nn.Linear(hidden_channels, hidden_channels)
        
        # 创建GraphTransformer层的列表
        self.transformer_layers = nn.ModuleList([
            GraphTransformerLayer(hidden_channels, num_heads, dropout)
            for _ in range(layers)
        ])
        
        self._init_weights()

    def _init_weights(self):
        """初始化模型权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.1)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x, edge_index):
        """前向传播
        
        参数:
            x (Tensor): 节点特征矩阵，形状为[num_nodes, in_channels]
            edge_index (Tensor): 边索引，形状为[2, num_edges]
            
        返回:
            Tensor: 处理后的节点特征，形状为[num_nodes, hidden_channels]
        """
        # 输入特征投影
        x = self.input_proj(x)
        
        # 依次应用每个Transformer层
        for layer in self.transformer_layers:
            x = layer(x, edge_index)
            
        return x


class GraphTransformerLayer(nn.Module):
    """Graph Transformer的单层实现
    
    实现基于图结构的自注意力机制，处理节点间的关系。
    
    参数:
        hidden_channels (int): 隐藏层维度
        num_heads (int): 注意力头数量，默认为4
        dropout (float): Dropout比率，默认为0.1
    """
    def __init__(self, hidden_channels, num_heads=4, dropout=0.1):
        super(GraphTransformerLayer, self).__init__()
        self.hidden_channels = hidden_channels
        self.num_heads = num_heads
        self.head_dim = hidden_channels // num_heads
        
        # 查询、键、值的线性投影层
        self.q_proj = nn.Linear(hidden_channels, hidden_channels)
        self.k_proj = nn.Linear(hidden_channels, hidden_channels)
        self.v_proj = nn.Linear(hidden_channels, hidden_channels)
        self.o_proj = nn.Linear(hidden_channels, hidden_channels)
        
        # 边特征编码器
        self.edge_encoder = nn.Sequential(
            nn.Linear(self.head_dim * 2, self.head_dim),
            nn.ReLU()
        )
        
        # 前馈神经网络
        self.ffn = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels * 2, hidden_channels)
        )
        
        # 归一化和dropout层
        self.norm1 = nn.Identity()
        self.norm2 = nn.Identity()
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, edge_index):
        """前向传播
        
        参数:
            x (Tensor): 节点特征矩阵，形状为[num_nodes, hidden_channels]
            edge_index (Tensor): 边索引，形状为[2, num_edges]
            
        返回:
            Tensor: 更新后的节点特征，形状为[num_nodes, hidden_channels]
        """
        # 残差连接
        residual = x 
        
        # 多头自注意力计算
        x = self.norm1(x)  # 归一化
        q = self.q_proj(x).view(-1, self.num_heads, self.head_dim)  # 查询投影
        k = self.k_proj(x).view(-1, self.num_heads, self.head_dim)  # 键投影
        v = self.v_proj(x).view(-1, self.num_heads, self.head_dim)  # 值投影
        
        # 提取源节点和目标节点
        src, dst = edge_index
        
        # 获取目标节点的查询和源节点的键
        q_i = q[dst]
        k_j = k[src]
        
        # 计算注意力分数和权重
        scale = self.head_dim ** -0.5  # 缩放因子
        scores = (q_i * k_j).sum(dim=-1) * scale
        attn_weights = F.softmax(scores, dim=0)
        attn_weights = self.dropout(attn_weights)

        # 应用注意力权重
        v_j = v[src]
        attn_output = v_j * attn_weights.unsqueeze(-1)

        # 聚合注意力输出
        out = torch.zeros_like(x).view(-1, self.num_heads, self.head_dim)
        out.index_add_(0, dst, attn_output)
        
        # 重塑输出并应用投影
        out = out.reshape(-1, self.hidden_channels)
        out = self.o_proj(out)
        out = self.dropout(out)
        
        # 第一个残差连接
        x = residual + out
        
        # 前馈网络和第二个残差连接
        residual = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = residual + x
        
        return x