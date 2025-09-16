import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool
from torch_geometric.data import Data, Batch



class Attention(nn.Module):
    """双向注意力机制
    
    实现两个特征集合之间的双向注意力交互。
    
    参数:
        dim (int): 特征维度
    """
    def __init__(self, dim):
        super(Attention, self).__init__()
        self.proj1 = nn.Linear(dim, 1)  # x1影响x2的注意力权重
        self.proj2 = nn.Linear(dim, 1)  # x2影响x1的注意力权重
        
    def forward(self, x1, x2):
        """前向传播
        
        参数:
            x1 (Tensor): 第一组特征
            x2 (Tensor): 第二组特征
            
        返回:
            Tensor: 融合后的特征
        """
        # 计算x1对x2的注意力
        energy1 = self.proj1(torch.abs(x1 - x2))
        attn1 = torch.sigmoid(energy1)
        out1 = x2 * attn1
        
        # 计算x2对x1的注意力
        energy2 = self.proj2(torch.abs(x2 - x1))
        attn2 = torch.sigmoid(energy2)
        out2 = x1 * attn2
        
        # 返回融合结果
        return out1 + out2


class MixPool(nn.Module):
    """混合池化模块
    
    结合均值池化和最大池化，使用可学习的权重参数。
    """
    def __init__(self):
        super(MixPool, self).__init__()
        self.alpha = nn.Parameter(torch.tensor(0.5))  # 可学习的混合参数
        
    def forward(self, x, batch):
        """前向传播
        
        参数:
            x (Tensor): 节点特征
            batch (Tensor): 批次指示向量
            
        返回:
            Tensor: 池化后的图特征
        """
        # 执行均值池化和最大池化
        x_mean = global_mean_pool(x, batch)
        x_max = global_max_pool(x, batch)
        
        # 动态调整混合权重
        alpha = torch.sigmoid(self.alpha)
        
        # 根据权重混合两种池化结果
        return alpha * x_mean + (1 - alpha) * x_max


class DualStreamAdaptiveFusion(nn.Module):#DAF
    """基于CAGN-GAT的自适应特征融合模块
    
    实现原始特征和交互特征的自适应融合。
    
    参数:
        hidden_dim (int): 隐藏层维度
    """
    def __init__(self, hidden_dim):
        super(DualStreamAdaptiveFusion, self).__init__()
        self.hidden_dim = hidden_dim
        
        # 原始特征处理流 (x1_pool, x2_pool, x3_pool)
        self.primary_stream = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim * 4),
            nn.LayerNorm(hidden_dim * 4),
            nn.GELU()
        )
        
        # 交互特征处理流 (x12, x13, x23)
        self.interaction_stream = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim * 4),
            nn.LayerNorm(hidden_dim * 4),
            nn.GELU()
        )
        
        # 自适应融合权重
        self.alpha = nn.Parameter(torch.tensor(0.5))
        
        # 特征增强层
        self.enhancement = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim * 4),
            nn.BatchNorm1d(hidden_dim * 4),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        
        # 初始化权重
        self._init_weights()
        
    def _init_weights(self):
        """初始化模型权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=1.0)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x):
        """前向传播
        
        参数:
            x (Tensor): 连接的特征向量 [batch_size, hidden_dim*6]
            
        返回:
            Tensor: 融合后的特征 [batch_size, hidden_dim*4]
        """
        # 分离原始特征和交互特征
        original_features = x[:, :self.hidden_dim*3]  # x1_pool, x2_pool, x3_pool
        interaction_features = x[:, self.hidden_dim*3:]  # x12, x13, x23
        
        # 双流处理
        h_primary = self.primary_stream(original_features)
        h_interaction = self.interaction_stream(interaction_features)
        
        # 自适应融合 (公式 12: h = α·h_primary + (1-α)·h_interaction)
        alpha = torch.sigmoid(self.alpha)
        fused_features = alpha * h_primary + (1 - alpha) * h_interaction
        
        # 特征增强
        enhanced_features = self.enhancement(fused_features)
        
        return enhanced_features