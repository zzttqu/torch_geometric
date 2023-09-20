import numpy as np
import networkx as nx
from matplotlib import pyplot as plt
import torch
from torch_geometric.data import Data
from torch.distributions import Categorical
from torch_geometric.data import Data
from GNNNet import GNNNet
from GNNAgent import Agent, PPOMemory
from envClass import EnvRun, StateCode
from torch.utils.tensorboard import SummaryWriter
import csv

if __name__ == "__main__":
    # # 写入一个csv文件
    # with open("./log.csv", "a", newline="") as csvfile:
    #     writer = csv.writer(csvfile)
    #     writer.writerow([0, 0, 0])
    # 读取用np读，写入用csv写
    data = np.genfromtxt("./log.csv", delimiter=",", skip_header=1)
    # 检测nparray有几个维度
    data.ndim
    init_step = int(1)
    print(data)
    raise SystemExit
    # 设置显示
    plt.rcParams["font.sans-serif"] = ["SimHei"]  # 显示中文标签
    plt.rcParams["axes.unicode_minus"] = False
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    np.set_printoptions(precision=3, suppress=True)
    torch.set_printoptions(precision=3, sci_mode=False)
    function_num = 6
    work_cell_num = 14
    batch_size = 4
    agent_reward = 0
    max_steps = 500
    total_step = init_step
    epoch_step = 0
    memory = PPOMemory(batch_size, work_cell_num, function_num, 4, 2, device)
    env = EnvRun(work_cell_num=work_cell_num, function_num=function_num, device=device)
    agent = Agent(
        work_cell_num,
        function_num,
        batch_size=batch_size,
        n_epochs=32,
        mini_batch_size=4,
    )
    # 如果不可视化节点就不用取返回值graph
    # 加入tensorboard
    writer = SummaryWriter(log_dir="logs")
    graph = env.build_edge()
    work_function = env.get_work_cell_functions()
    weight = torch.tensor([1] * work_cell_num, dtype=torch.float).squeeze()
    random_function = torch.tensor([row[0] for row in work_function]).squeeze()
    # 初始状态
    # raw = torch.tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    # 加载之前的
    # agent.load_model()
    obs_states, edge_index, reward, dones = env.get_obs()
    print(obs_states)
    # 添加计算图
    writer.add_graph(
        agent.network, input_to_model=[obs_states, edge_index], verbose=False
    )
    # 添加
    while total_step < init_step + max_steps:
        total_step += 1
        epoch_step += 1
        agent.network.eval()
        with torch.no_grad():
            raw, log_prob = agent.get_action(obs_states, edge_index)
            value = agent.get_value(obs_states, edge_index)
        env.update_all(raw.cpu())
        obs_states, edge_index, reward, dones = env.get_obs()
        agent_reward += reward
        memory.remember(obs_states, value, agent_reward, dones, raw, log_prob)
        # 如果记忆数量等于batch_size就学习
        if memory.count == batch_size:
            agent.network.train()
            loss = agent.learn(
                memory,
                last_node_state=obs_states,
                last_done=dones,
                edge_index=edge_index,
            )
            writer.add_scalar("loss", loss, total_step)
        if dones == 1:
            print("=======")
            print(agent_reward)
            writer.add_scalar("reward", agent_reward, total_step)
            with open("./log.csv", "a", newline="") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([total_step, agent_reward, loss])
            epoch_step = 0
            agent.network.save_model("model_" + str(total_step) + ".pth")
            # print(env.center_list[2].product_num)
            env.reset()
            agent_reward = 0
        # if step % 50 == 0:
        #     # print(agent_reward, obs_states, raw)
        #     agent_reward = 0
        #     env.reset()
        #     obs_states, edge_index, reward, dones = env.get_obs()

    # 神经网络要输出每个工作站的工作，功能和传输与否

    # 可视化
    node_states = nx.get_node_attributes(graph, "state")
    node_function = nx.get_node_attributes(graph, "function")
    nodes = nx.nodes(graph)
    edges = nx.edges(graph)
    node_labels = {}
    edge_labels = {}
    for node in nodes:
        # 这里只用\n就可以换行了
        node_labels[
            node
        ] = f"{node}节点：\n 状态：{node_states[node]} \n 功能：{node_function[node]}"

    # print(node_labels)
    pos = nx.spring_layout(graph)
    nx.draw_networkx_nodes(graph, pos)
    nx.draw_networkx_labels(graph, pos, node_labels)
    # nx.draw_networkx_edges(graph, pos, connectionstyle="arc3,rad=0.2")
    nx.draw_networkx_edges(graph, pos)

    plt.show()
