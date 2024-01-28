from datetime import datetime
from torch import nn
import torch
from GNNAgent import Agent
from envClass import EnvRun
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.nn import aggr
from loguru import logger


class Test(nn.Module):
    def __init__(self):
        super(Test, self).__init__()
        self.embedding1 = nn.Embedding(10, 3)
        self.embedding2 = nn.EmbeddingBag(10, 3, sparse=True)

    def forward(self, data, offsets):
        x = self.embedding1(data)
        x2 = self.embedding2(data, offsets)
        return x, x2


if __name__ == "__main__":
    test = Test()
    indices = torch.tensor([0, 1.1, 2, 3])
    offsets = torch.tensor([0, 1, 2, 3])
    x = test(indices, offsets)
    logger.info(x)
