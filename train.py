import numpy as np
import networkx as nx
from matplotlib import pyplot as plt
import torch
from GNNAgent import Agent, PPOMemory
from envClass import EnvRun, select_functions
from torch.utils.tensorboard import SummaryWriter
import csv
from datetime import datetime

if __name__ == "__main__":
    # # 写入一个csv文件
    # with open("./log.csv", "a", newline="") as csvfile:
    #     writer = csv.writer(csvfile)
    #     writer.writerow([0, 0, 0])
    # 读取用np读，写入用csv写
    try:
        data = np.genfromtxt("./log.csv", delimiter=",", skip_header=1)
    except:
        data = 0
    # 检测nparray有几个维度
    if data == 0:
        init_step = 0
    else:
        init_step = int(data[-1][0])
    # 设置显示
    # print(select_functions(0, 6, 14))
    # raise SystemExit
    plt.rcParams["font.sans-serif"] = ["SimHei"]  # 显示中文标签
    plt.rcParams["axes.unicode_minus"] = False
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    np.set_printoptions(precision=3, suppress=True)
    torch.set_printoptions(precision=3, sci_mode=False)
    function_num = 6
    work_cell_num = 15
    batch_size = 64
    agent_reward = 0
    max_steps = 1000
    max_episode_step = 32
    total_step = init_step
    epoch_step = 0
    episode_num = 0
    learn_num = 0

    env = EnvRun(work_cell_num=work_cell_num, function_num=function_num, device=device)
    agent = Agent(
        work_cell_num,
        function_num,
        batch_size=batch_size,
        n_epochs=32,
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
    agent.load_model("last_model.pth")
    obs_states, edge_index, reward, dones = env.get_obs()
    memory = PPOMemory(
        batch_size, work_cell_num, function_num, edge_index.shape[1], 4, 2, device
    )
    init_time = datetime.now()
    print(f"模型加载完成，环境初始化完成，当前时间{init_time.strftime('%Y-%m-%d %H:%M:%S')}")
    now_time = datetime.now()
    # print(obs_states)
    # 添加计算图
    # writer.add_graph(
    #    agent.network, input_to_model=[obs_states, edge_index], verbose=False
    # )
    # 添加
    while total_step < init_step + max_steps:
        total_step += 1
        epoch_step += 1
        agent.network.eval()
        with torch.no_grad():
            raw, log_prob = agent.get_action(obs_states, edge_index)
            value = agent.get_value(obs_states, edge_index)
        # 这个raw少了
        env.update_all(raw.cpu())
        obs_states, edge_index, reward, dones = env.get_obs()
        agent_reward += reward
        if (epoch_step >= max_episode_step) and (dones != 1):
            dones = 1
            agent_reward -= 5

        memory.remember(
            obs_states, edge_index, value, agent_reward, dones, raw, log_prob
        )
        # 如果记忆数量等于batch_size就学习
        if memory.count == batch_size:
            learn_num += 1
            agent.network.train()
            agent.learn(
                memory,
                last_node_state=obs_states,
                last_done=dones,
                edge_index=edge_index,
                writer=writer,
            )
            learn_time = (datetime.now() - now_time).seconds
            print(f"第{learn_num}次学习，学习用时：{learn_time}秒")
            now_time = datetime.now()
            # writer.add_scalar("loss", loss, total_step)
        if dones == 1:
            print("=================")
            episode_num += 1
            print(f"总步数：{total_step}，本次循环步数为：{epoch_step}，奖励为{agent_reward:.3f}")
            writer.add_scalar("reward", agent_reward, total_step)
            writer.add_scalars(
                "products",
                {
                    f"产品{i}": env.total_products[i]
                    for i in range(1, env.total_products.shape[0])
                },
                total_step,
            )
            with open("./log.csv", "a", newline="") as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow([total_step, f"{agent_reward:.3f}"])
            epoch_step = 0
            env.reset()
            agent_reward = 0
        if total_step % 500 == 0:
            agent.save_model("model_" + str(total_step) + ".pth")

    # 神经网络要输出每个工作站的工作，功能和传输与否
    agent.save_model("last_model.pth")
    total_time = (datetime.now() - init_time).seconds // 60
    print(f"总计用时：{total_time}，运行{total_step}步，学习{learn_num}次")

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

    # plt.show()
