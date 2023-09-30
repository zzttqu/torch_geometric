from torch_geometric.nn import (
    GCNConv,
    SAGEConv,
    # 池化，减少节点数量
    TopKPooling,
    # 聚合层
    SoftmaxAggregation,
    GATConv,
    # 骨干网络
    GATv2Conv,
    GraphSAGE,
    Linear,
    # embedding，增强泛化性
    MetaPath2Vec,
    # 转为异构图
    to_hetero,
)
from torch_geometric.datasets import OGB_MAG
import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.data import Data, Batch, HeteroData
from torch_geometric.loader import DataLoader
import torch_geometric.transforms as T

embedding = MetaPath2Vec(
    edge_index_dict=edge_dic,
    embedding_dim=20,
    walk_length=3,
    context_size=2,
    metapath=meta,
)
embedding.state_dict

class GNNNet(nn.Module):
    def __init__(self, edge_dic: dict, action_dim: int, meta, action_choice=2):
        super(GNNNet, self).__init__()
        # 将节点映射为一个四维向量

        self.conv1 = GATConv(-1, 64, add_self_loops=False)
        self.conv2 = GATConv(64, 128, add_self_loops=False)
        # self.pool2 = TopKPooling(128, ratio=0.8)
        # self.agg1 = SoftmaxAggregation(learn=True)
        # self.lerelu = nn.LeakyReLU()
        self.relu = nn.ReLU()
        self.lin1 = nn.Linear(128, 64)
        self.lin11 = Linear(-1, 64)
        self.lin2 = nn.Linear(64, 64)
        self.lin3 = nn.Linear(64, action_dim * action_choice)
        self.linV = nn.Linear(64, 1)

        # self.bn1 = nn.BatchNorm1d(128)
        # self.bn2 = nn.BatchNorm1d(64)

    def forward(self, x: dict, edge_index):
        """
        前向传播
        """
        # x = self.embedding(x)
        # x = x.squeeze(1)
        x = self.embedding(x)
        x = self.conv1(x, edge_index).relu()

        # x, edge_index, _, batch, _, _ = self.pool1(x, edge_index)
        # x1 = global_mean_pool(x, None)
        x = self.conv2(x, edge_index).relu()
        # x, edge_index, _, batch, _, _ = self.pool2(x, edge_index)
        # x = global_mean_pool(x, batch)
        # x = x1 + x2
        x = self.lin1(x)

        x = self.lin2(x)
        # 两个输出
        value = self.linV(x).mean()
        x = torch.tanh(self.lin11(x))
        return x, value

    def save_model(self, name):
        """
        保存模型
        """
        torch.save(self.state_dict(), name)

    def load_model(self, name):
        """
        加载模型
        """
        pretrained_model = torch.load(name)
        pretrained_model = {
            key: value
            for key, value in pretrained_model.items()
            if (key in self.state_dict() and "lin3" not in key)
        }
        self.load_state_dict(pretrained_model, strict=False)
