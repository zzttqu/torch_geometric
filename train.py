import random
from enum import Enum
import numpy as np
import networkx as nx
from matplotlib import pyplot as plt
import torch
from torch_geometric.data import Data

from torch.distributions import Categorical
from torch_geometric.data import Data
from GNNNet import GNNNet
from envClass import EnvRun, StateCode

if __name__ == '__main__':
    # 设置显示
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文标签
    plt.rcParams['axes.unicode_minus'] = False
    np.set_printoptions(precision=3, suppress=True)
    torch.set_printoptions(precision=3, sci_mode=False)
    function_num = 3
    work_cell_num = 6
    env = EnvRun(1, work_cell_num=work_cell_num, function_num=function_num)
    graph = env.build_edge()
    obs_states, edge_index, reward, dones = env.get_obs()
    work_function = env.get_work_cell_functions()
    weight = torch.tensor([1] * work_cell_num, dtype=torch.float)
    random_function = torch.tensor([row[0] for row in work_function])
    # 设置任务计算生产
    env.update_all(random_function + weight)
    # 神经网络要输出每个工作站的工作，功能和传输与否

    # 可视化
    node_states = nx.get_node_attributes(graph, 'state')
    node_function = nx.get_node_attributes(graph, 'function')
    nodes = nx.nodes(graph)
    edges = nx.edges(graph)
    node_labels = {}
    edge_labels = {}
    for node in nodes:
        # 这里只用\n就可以换行了
        node_labels[node] = f'{node}节点：\n 状态：{node_states[node]} \n 功能：{node_function[node]}'

    # print(node_labels)
    pos = nx.spring_layout(graph)
    nx.draw_networkx_nodes(graph, pos)
    nx.draw_networkx_labels(graph, pos, node_labels)
    # nx.draw_networkx_edges(graph, pos, connectionstyle="arc3,rad=0.2")
    nx.draw_networkx_edges(graph, pos)

    net = GNNNet(node_num=6, state_dim=4, action_dim=4).double()
    actions = net(data)
    actions = actions.reshape((12, 2)).squeeze()
    # split_actions = torch.split(actions, 4, dim=1)

    # 第一项是功能动作，第二项是是否接受上一级运输
    actions_dist = Categorical(logits=actions)

    material_dist = [Categorical(logits=actions[6:, :])]
    actions = actions_dist.sample()
    actions = actions[:6]

    materials = torch.stack([dist.sample() for dist in material_dist]).squeeze()
    env.update_work_cell([0, 1, 2, 3, 4, 5], actions)
    env.update_centers(materials)
    obs_state, obs_edge_index, reward = env.get_obs()
    data = Data(x=obs_state, edge_index=obs_edge_index)
    # print(obs_state)

    # plt.show()
