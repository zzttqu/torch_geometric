import torch
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, Linear, Node2Vec
import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.datasets import KarateClub
from torch_geometric.utils import to_networkx
from torch import nn

# 节点属性
x = torch.tensor([[2, 1], [5, 6], [3, 7], [12, 0]], dtype=torch.float)
# 节点分类
y = torch.tensor([0, 1, 0, 1])
# 边数据
nn.Linear(10, 10)
# 0连接了1,1连接了0,2连接了1,0连接了3,3连接了2
edge_index = torch.tensor([[0, 1, 2, 0, 3],
                           [1, 0, 1, 3, 2]])
graph = nx.Graph()
graph.add_edges_from(edge_index.T)
data = Data(x=x, y=y, edge_index=edge_index)
dataset = KarateClub()
print(f'{dataset.len()}')
G = to_networkx(dataset[0])
# 可视化
nx.draw_networkx(G, pos=nx.spring_layout(G, seed=42), with_labels=False, node_color=dataset[0].y)

print(data)


class GCN(torch.nn.Module):
    def __init__(self):
        super(GCN, self).__init__()
        torch.manual_seed(123)
        self.conv1 = GCNConv(dataset.num_features, 4)
        self.conv2 = GCNConv(4, 4)
        self.classifier = nn.Linear(4, dataset.num_classes)

    def forward(self, x, edge_index):
        h = self.conv1(x, edge_index)
        h = h.tanh()
        h = self.conv2(h, edge_index)
        h = h.tanh()
        out = self.classifier(h)
        return out, h


model = GCN()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
print(model)


def visualize_embedding(h, color):
    plt.figure(figsize=(7, 7))
    plt.xticks([])
    plt.yticks([])
    h = h.detach().cpu().numpy()
    plt.scatter(h[:, 0], h[:, 1], s=140, c=color, cmap='Set2')
    plt.show()


def train(data):
    optimizer.zero_grad()
    out, h = model(data.x, data.edge_index)
    loss = criterion(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss, h


for i in range(1000):
    loss, h = train(dataset)
visualize_embedding(h, dataset[0].y)
