import torch
from torch_geometric.nn import GCNConv
from torch import nn
import torchvision.models as models
#import gym
from envClass import AGVCell, WorkCell, StateCode


class WorkShopEnv(gym.Env):
    def __init__(self, num_jobs: int):
        self.num_jobs = num_jobs
        self.reset()


class GNNNet(nn.Module):
    def __init__(self, num_node, hidden_dim, out_dim):
        super(GNNNet, self).__init__()
        self.gnn1 = GCNConv(in_channels=num_node, out_channels=hidden_dim)
        self.gnn2 = GCNConv(in_channels=hidden_dim, out_channels=out_dim)

    def forward(self, x):
        x = self.gnn1(x)
        x = self.gnn2(x)


torch.optim.NAdam
