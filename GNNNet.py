from torch_geometric.nn import (
    GCNConv,
    SAGEConv,
    TopKPooling,
    global_mean_pool,
    Node2Vec,
)
import torch
from torch import nn
from torch_geometric.data import Data
import torch.nn.functional as F


class GNNNet(nn.Module):
    def __init__(self, state_dim: int, node_num: int, action_dim: int, action_choice=2):
        super(GNNNet, self).__init__()
        # 将节点映射为一个四维向量
        # self.embedding = nn.Embedding(num_embeddings=1000, embedding_dim=state_dim)
        self.conv1 = GCNConv(state_dim, 128)
        self.pool1 = TopKPooling(128, ratio=0.8)
        self.conv2 = GCNConv(128, 128)
        self.pool2 = TopKPooling(128, ratio=0.8)
        self.lin1 = nn.Linear(128, 128)
        self.lin2 = nn.Linear(128, 64)
        self.lin3 = nn.Linear(64, action_dim * node_num * action_choice)
        self.linV = nn.Linear(64, 1)
        # self.bn1 = nn.BatchNorm1d(128)
        # self.bn2 = nn.BatchNorm1d(64)

    def forward(self, x, edge_index):
        """
        前向传播
        """
        # x, edge_index, batch = data.x, data.edge_index, data.batch
        # x = self.embedding(x)
        # x = x.squeeze(1)
        # print(x, edge_index, edge_attr)
        # print(x.dtype)
        x = F.relu(self.conv1(x, edge_index))
        # print(x.dtype)
        x, edge_index, _, batch, _, _ = self.pool1(x, edge_index)
        x1 = global_mean_pool(x, None)
        x = F.relu(self.conv2(x, edge_index))
        x, edge_index, _, batch, _, _ = self.pool2(x, edge_index)
        x2 = global_mean_pool(x, None)
        x = x1 + x2
        x = self.lin1(x)
        x = self.lin2(x)
        # 两个输出
        value = self.linV(x)
        x = torch.tanh(self.lin3(x))
        return x, value

    def save_model(self):
        """
        保存模型
        """
        torch.save(self.state_dict(), "./model/model.pth")

    def load_model(self):
        """
        加载模型
        """
        self.load_state_dict(torch.load("./model/model.pth"))
