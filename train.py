import numpy as np
import networkx as nx
from matplotlib import pyplot as plt
import torch
from GNNAgent import Agent, PPOMemory
from torch_geometric.data import Data, Batch, HeteroData
from envClass import EnvRun, select_functions
from torch.utils.tensorboard.writer import SummaryWriter
import csv
from datetime import datetime


def show(graph):
    # 可视化
    node_states = nx.get_node_attributes(graph, "state")
    node_function = nx.get_node_attributes(graph, "function")
    nodes = nx.nodes(graph)
    edges = nx.edges(graph)
    node_labels = {}
    edge_labels = {}
    pos = {}
    for node in nodes:
        # 这里只用\n就可以换行了
        node_labels[
            node
        ] = f"{node}节点：\n 状态：{node_states[node]} \n 功能：{node_function[node]}"
        pos[node] = (node_function, node)

    # print(node_labels)
    pos = nx.spring_layout(graph)
    nx.draw_networkx_nodes(graph, pos)
    nx.draw_networkx_labels(graph, pos, node_labels)
    # nx.draw_networkx_edges(graph, pos, connectionstyle="arc3,rad=0.2")
    nx.draw_networkx_edges(graph, pos)
    plt.show()


if __name__ == "__main__":
    # # 写入一个csv文件
    # with open("./log.csv", "a", newline="") as csvfile:
    #     writer = csv.writer(csvfile)
    #     writer.writerow([0, 0, 0])
    # 读取用np读，写入用csv写
    try:
        data = np.genfromtxt("./log.csv", delimiter=",", skip_header=1)
        init_step = int(data[-1, 0]) if data.ndim > 1 else 0
    except (IOError, ValueError):
        # 处理文件不存在或数据不可读的情况
        init_step = 0
    # 设置显示
    plt.rcParams["font.sans-serif"] = ["SimHei"]  # 显示中文标签
    plt.rcParams["axes.unicode_minus"] = False
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    np.set_printoptions(precision=3, suppress=True)
    torch.set_printoptions(precision=3, sci_mode=False)
    # 神奇trick
    torch.manual_seed(3407)
    function_num = 5
    work_center_num = 10
    batch_size = 64

    total_step = init_step
    max_steps = 64 * 4
    episode_step_max = 64
    product_goal = 200
    n_epochs = 16

    episode_num = 0
    learn_num = 0

    env = EnvRun(
        work_center_num=work_center_num,
        fun_per_center=2,
        function_num=function_num,
        device=device,
        episode_step_max=episode_step_max,
        product_goal=product_goal,
    )
    # 如果不可视化节点就不用取返回值graph
    # 加入tensorboard
    writer = SummaryWriter(log_dir="logs/train")
    edge_index = env.build_edge()

    # 加载之前的

    obs_states, edge_index, reward, dones, _ = env.get_obs()
    print(obs_states)
    # print(f"初始化状态为{obs_states}")
    # print(f"初始化边为{edge_index}")
    print(f"加工能力为{env.product_capacity}")
    hetero_data = HeteroData()
    # 节点信息
    for key, _value in obs_states.items():
        hetero_data[key].x = _value
        # 边信息
    for key, _value in edge_index.items():
        node1, node2 = key.split("_to_")
        hetero_data[(f"{node1}", f"{key}", f"{node2}")].edge_index = _value
    agent = Agent(
        batch_size=batch_size,
        n_epochs=n_epochs,
        init_data=hetero_data,
    )

    agent.load_model("last_model.pth")
    memory = PPOMemory(
        batch_size,
        device,
    )

    init_time = datetime.now()
    print(f"模型加载完成，环境初始化完成，当前时间{init_time.strftime('%Y-%m-%d %H:%M:%S')}")
    now_time = datetime.now()

    # 添加计算图

    agent.network(obs_states, edge_index)

    writer.add_graph(
        agent.network,
        input_to_model=[obs_states, edge_index],
        verbose=False,
        use_strict_trace=False,
    )
    # 添加
    # show()
    while total_step < init_step + max_steps:
        total_step += 1
        agent.network.eval()
        with torch.no_grad():
            # raw是一个2*节点数量
            raw, log_prob = agent.get_action(obs_states, edge_index)
            value = agent.get_value(obs_states, edge_index)

        # 这个raw因为是字典，这里变了之后会影响get action中的raw
        # 后来还是改为了直接的tensor
        # for key, _value in raw.items():
        #    raw[key] = _value.cpu()
        assert isinstance(raw, torch.Tensor), "raw 不是tensor"
        assert raw.device != "cpu", "raw 不在cpu中"
        env.update_all(raw.cpu())
        # 所以需要搬回cuda中
        # for key, _value in raw.items():
        #    raw[key] = _value.to(device)
        obs_states, edge_index, reward, dones, episode_step = env.get_obs()
        env.show_graph()
        # raise SystemExit
        writer.add_scalars(
            "step/products",
            {
                f"产品{i}": env.total_products[i]
                for i in range(0, env.total_products.shape[0])
            },
            total_step,
        )
        writer.add_scalar("step/reward", reward, total_step)
        memory.remember(obs_states, edge_index, value, reward, dones, raw, log_prob)
        # 如果记忆数量等于batch_size就学习
        if memory.count == batch_size:
            learn_num += 1
            agent.network.train()
            loss = agent.learn(
                memory,
                last_node_state=obs_states,
                last_done=dones,
                edge_index=edge_index,
            )
            learn_time = (datetime.now() - now_time).seconds
            print(f"第{learn_num}次学习，学习用时：{learn_time}秒")
            agent.save_model("last_model.pth")
            now_time = datetime.now()
            writer.add_scalar("loss", loss, total_step)
        if dones == 1:
            print("=================")
            print(f"总步数：{total_step}，本次循环步数为：{episode_step}，奖励为{ reward:.3f}")
            writer.add_scalar("reward", reward, total_step)
            with open("./log.csv", "a", newline="") as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow([total_step, f"{reward:.3f}"])
            env.reset()
        if total_step % 500 == 0:
            agent.save_model("model_" + str(total_step) + ".pth")

    # 神经网络要输出每个工作站的工作，功能和传输与否
    agent.save_model("last_model.pth")
    total_time = (datetime.now() - init_time).seconds // 60
    with open("./log.csv", "a", newline="") as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow([total_step, f"{reward:.3f}"])
    print(f"总计用时：{total_time}分钟，运行{total_step}步，学习{learn_num}次")
