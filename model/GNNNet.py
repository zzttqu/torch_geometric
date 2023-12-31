from typing import Dict, Tuple
from torch_geometric.nn import (
    GCNConv,
    SAGEConv,
    # 池化，减少节点数量
    TopKPooling,
    # 聚合层
    SoftmaxAggregation,
    # 骨干网络
    GATConv,
    GATv2Conv,
    GraphSAGE,
    Linear,
    HANConv,
    HeteroLinear,
    HeteroDictLinear,
    # embedding，增强泛化性，只是在节点没有信息的时候可以用
    MetaPath2Vec,
    # 转为异构图
    to_hetero,
)
import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from loguru import logger

"""embedding = MetaPath2Vec(
    edge_index_dict=edge_dic,
    embedding_dim=20,
    walk_length=3,
    context_size=2,
    metapath=meta,
)
embedding.state_dict"""


# dataset = OGB_MAG(root="./data")
# data = dataset[0]
# print(data.num_node_features)

# raise SystemExit


class GNNNet(nn.Module):
    def __init__(self, action_dim: int, action_choice=2):
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

    def forward(self, x, edge_index):
        """
        前向传播
        """
        # x = self.embedding(x)
        # x = x.squeeze(1)
        # x = self.embedding(x)
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


class HGTNet(nn.Module):
    def __init__(
            self,
            data: HeteroData,
            hidden_channels=64,
            num_layers=2,
            action_choice=2,
    ):
        super().__init__()
        # 将节点映射为一个四维向量
        self.encoders = torch.nn.ModuleDict()
        node_features_dict = data.num_node_features
        for node_type in data.node_types:
            self.encoders[f"{node_type}_linear"] = nn.Linear(
                data.num_node_features[node_type], hidden_channels
            )
        self.conv_list = torch.nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                conv = HANConv(node_features_dict, hidden_channels, data.metadata(), heads=2)
            else:
                conv = HANConv(hidden_channels, hidden_channels, data.metadata(), heads=2)
            self.conv_list.append(conv)
        self.lin0 = nn.Linear(hidden_channels, hidden_channels)
        # 这里先写死了吧，应该需要根据需求进行设置
        # 工作中心一个动作，动作空间为工作单元数量+1，表示启动哪个工作单元，工作单元一个动作，是否接收
        self.linCenter = nn.Linear(hidden_channels, 3)
        self.linCell = nn.Linear(hidden_channels, 2)
        self.linV = nn.Linear(hidden_channels, 1)
        # self.pool = SoftmaxAggregation(channels=hidden_channels)

        # self.bn1 = nn.BatchNorm1d(128)
        # self.bn2 = nn.BatchNorm1d(64)

    def forward(
            self,
            x_dict: Dict[str, torch.Tensor],
            edge_index_dict: Dict[str, torch.Tensor],
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """
        前向传播，tensorboard不支持tuple类型的输入，需要单独的字符串作为字典索引
        """
        norm_edge_index_dict = {}
        for key, _value in edge_index_dict.items():
            node1, node2 = key.split("_to_")
            norm_edge_index_dict[f"{node1}", f"{key}", f"{node2}"] = _value
        # 根据node type分别传播，这里由于改了inputdim，还不能直接去掉encoder层
        # 目前不用encoder层
        # x_dict = {
        #     node_type: F.leaky_relu(self.encoders[f"{node_type}_linear"](x))
        #     for node_type, x in x_dict.items()
        # }
        # x_dict = {
        #     node_type: F.leaky_relu(self.encoders[f"{node_type}_linear"](x))
        #     for node_type, x in x_dict.items()
        # }

        for conv in self.conv_list:
            x_dict = conv(x_dict, norm_edge_index_dict)
        # 两个输出，一个需要连接所有节点的特征然后输出一个value
        full_x = torch.cat([x for x in x_dict.values()], dim=0)

        value = self.linV(full_x).mean()
        # 另一个需要根据每个节点用异质图线性层输出成一个dict
        # dict无法直接用tanh激活，还需要for
        x_dict = {
            node_type: F.leaky_relu(self.lin0(x)) for node_type, x in x_dict.items()
        }
        action = {}
        for node_type, x in x_dict.items():
            if node_type == "center":
                action[node_type] = torch.tanh(self.linCenter(x))
            elif node_type == "cell":
                action[node_type] = torch.tanh(self.linCell(x))
        # action = {
        #     node_type: torch.tanh(self.linOut(x)) for node_type, x in x_dict.items()
        # }
        return action, value

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
