from torch_geometric.nn import (
    GCNConv,
    SAGEConv,
    TopKPooling,
    global_mean_pool,
    GATConv,
    GATv2Conv,
    GraphSAGE
    
)
import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader


class GNNNet(nn.Module):
    def __init__(self, state_dim: int, node_num: int, action_dim: int, action_choice=2):
        super(GNNNet, self).__init__()
        # 将节点映射为一个四维向量
        # self.embedding = nn.Embedding(num_embeddings=1000, embedding_dim=state_dim)
        self.conv1 = GATv2Conv(state_dim, 128)
        self.pool1 = TopKPooling(128, ratio=0.8)
        self.conv2 = GATv2Conv(128, 128)
        self.pool2 = TopKPooling(128, ratio=0.8)
        self.lerelu = nn.LeakyReLU()
        self.relu = nn.ReLU()
        self.lin1 = nn.Linear(128, 128)
        self.lin2 = nn.Linear(128, 64)
        self.lin3 = nn.Linear(64, action_dim * action_choice)
        self.linV = nn.Linear(64, 1)

        # self.bn1 = nn.BatchNorm1d(128)
        # self.bn2 = nn.BatchNorm1d(64)

    def forward(self, data):
        """
        前向传播，data可以是batch，也可以是data，两种格式
        """
        x, edge_index, batch = data.x, data.edge_index, data.batch
        # x = self.embedding(x)
        # x = x.squeeze(1)
        x = self.relu(self.conv1(x, edge_index))
        # x, edge_index, _, batch, _, _ = self.pool1(x, edge_index)
        # x1 = global_mean_pool(x, None)
        x = self.relu(self.conv2(x, edge_index))
        # x, edge_index, _, batch, _, _ = self.pool2(x, edge_index)
        # x = global_mean_pool(x, batch)
        # x = x1 + x2
        x = self.lin1(x)
        x = self.lin2(x)
        # 两个输出
        value = self.linV(x).mean()
        x = torch.tanh(self.lin3(x))
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
