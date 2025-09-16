"""
MGTAF模型实现
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from layers.module import Attention, MixPool, DualStreamAdaptiveFusion
from layers.transformer import GraphTransformer, GCN

class GCN_Transformer(nn.Module):
    """GCN-Transformer模型
    
    参数:
        args: 包含模型配置的参数对象，主要使用:
            - input_dim: 输入特征维度
            - num_classes: 输出类别数
    """
    def __init__(self, args):
        super(GCN_Transformer, self).__init__()
        
        # 配置参数
        self.input_dim = args.input_dim  # 输入特征维度
        self.hidden_dim = 32  # 隐藏层维度
        self.num_classes = args.num_classes  # 输出类别数
        self.l2_reg = 1e-5  # L2正则化系数
        
        # GCN特征提取器
        self.gcn = GCN(self.input_dim, self.hidden_dim)
        
        # Graph Transformer层
        self.graph_transformer = GraphTransformer(
            in_channels=self.hidden_dim,
            hidden_channels=self.hidden_dim,
            num_heads=2,
            dropout=0.4,
            layers=1
        )
        
        # 注意力和池化模块
        self.attention = Attention(self.hidden_dim)
        self.pool = MixPool()
        
        # 特征融合模块
        self.fusion = DualStreamAdaptiveFusion(self.hidden_dim)
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_dim * 4, self.hidden_dim * 2),
            nn.BatchNorm1d(self.hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(self.hidden_dim * 2, self.num_classes)
        )
        
        # 初始化模型权重
        self._init_weights()
        
    def _init_weights(self):
        """初始化模型权重"""
        # 初始化线性层、BN层和GCN层
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, GCNConv) and hasattr(m, 'lin'):
                nn.init.xavier_uniform_(m.lin.weight, gain=1.414)
        
        # 初始化Transformer层
        if hasattr(self, 'graph_transformer') and hasattr(self.graph_transformer, 'transformer_layers'):
            for layer in self.graph_transformer.transformer_layers:
                for proj in [layer.q_proj, layer.k_proj, layer.v_proj, layer.o_proj]:
                    nn.init.normal_(proj.weight, mean=0.0, std=0.02)
        
        # 针对不同类型的fusion处理
        if hasattr(self, 'fusion'):
            if isinstance(self.fusion, nn.Sequential) and len(self.fusion) > 0:
                nn.init.xavier_uniform_(self.fusion[0].weight, gain=1.0)
        
        # 初始化分类器
        if hasattr(self, 'classifier') and len(self.classifier) > 0:
            first_layer = 0
            last_layer = len(self.classifier) - 1
            
            # 找到第一个线性层
            while not isinstance(self.classifier[first_layer], nn.Linear) and first_layer < last_layer:
                first_layer += 1
            
            # 找到最后一个线性层
            while not isinstance(self.classifier[last_layer], nn.Linear) and last_layer > 0:
                last_layer -= 1
            
            # 初始化第一个和最后一个线性层
            if isinstance(self.classifier[first_layer], nn.Linear):
                nn.init.xavier_uniform_(self.classifier[first_layer].weight, gain=1.0)
            if isinstance(self.classifier[last_layer], nn.Linear):
                nn.init.xavier_uniform_(self.classifier[last_layer].weight, gain=1.0)
    
    def forward(self, data):
        """前向传播
        
        参数:
            data: 包含三个图的PyG数据对象
            
        返回:
            Tensor: 模型输出的对数概率
        """
        # 提取三个图的节点特征和边索引
        x1, edge_index1 = data.x1, data.edge_index1
        x2, edge_index2 = data.x2, data.edge_index2
        x3, edge_index3 = data.x3, data.edge_index3
        
        # 通过GCN进行特征提取
        x1 = self.gcn(x1, edge_index1)
        x2 = self.gcn(x2, edge_index2)
        x3 = self.gcn(x3, edge_index3)
        
        # 通过Transformer进行特征增强
        x1 = self.graph_transformer(x1, edge_index1)
        x2 = self.graph_transformer(x2, edge_index2)
        x3 = self.graph_transformer(x3, edge_index3)
        
        # 获取批次索引
        batch1, batch2, batch3 = data.x1_batch, data.x2_batch, data.x3_batch
        
        # 对三个图进行混合池化
        x1_pool = self.pool(x1, batch1)  # [batch_size, hidden_dim]
        x2_pool = self.pool(x2, batch2)  # [batch_size, hidden_dim]
        x3_pool = self.pool(x3, batch3)  # [batch_size, hidden_dim]
        
        # 计算图间注意力交互
        x12 = self.attention(x1_pool, x2_pool)  # [batch_size, hidden_dim]
        x13 = self.attention(x1_pool, x3_pool)  # [batch_size, hidden_dim]
        x23 = self.attention(x2_pool, x3_pool)  # [batch_size, hidden_dim]
        
        # 特征拼接
        fusion_feat = torch.cat([
            x1_pool, x2_pool, x3_pool,  # 原始池化特征
            x12, x13, x23                # 交互特征
        ], dim=1)  # [batch_size, hidden_dim*6]
        
        # 特征融合
        fusion_feat = self.fusion(fusion_feat)  # [batch_size, hidden_dim*4]
        
        # 分类
        logits = self.classifier(fusion_feat)  # [batch_size, num_classes]
        
        # 计算L2正则化损失
        l2_loss = 0
        for param in self.parameters():
            l2_loss += torch.norm(param, 2)
        
        # 保存正则化损失供训练使用
        self.reg_loss = self.l2_reg * l2_loss
        
        # 返回对数概率
        return F.log_softmax(logits, dim=1)