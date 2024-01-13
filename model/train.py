import sys
from datetime import datetime

import numpy as np
import torch
from envClass import EnvRun
from matplotlib import pyplot as plt
from torch.utils.tensorboard.writer import SummaryWriter
from loguru import logger
from torch_geometric.data import HeteroData
from model.GNNAgent import Agent
from model.PPOMemory import PPOMemory
import cProfile


def main(max_steps):
    # # 写入一个csv文件
    # with open("./log.csv", "a", newline="") as csvfile:
    #     writer = csv.writer(csvfile)
    #     writer.writerow([0, 0, 0])
    # 读取用np读，写入用csv写
    order = torch.tensor([100, 600, 200], dtype=torch.int)
    # 这里应该是对各个工作单元进行配置了60个center
    work_center_init_func = torch.tensor([[3, 3, 10],
                                          [2, 2, 6],
                                          [4, 5, 0],
                                          [3, 0, 12],
                                          [2, 3, 5]], dtype=torch.int)

    speed_list = torch.tensor([[5, 10, 15, 20, 12], [8, 12, 18, torch.nan, 12], [3, 6, torch.nan, 10, 8]]).T
    init_step = 0
    # 设置
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # 神奇trick
    torch.manual_seed(3407)
    batch_size = 128

    total_step = init_step
    max_steps = max_steps
    episode_step_max = 256
    n_epochs = 12
    learn_num = 0

    env = EnvRun(
        order=order,
        speed_list=speed_list,
        work_center_init_func=work_center_init_func,
        device=device,
        expected_step=80,
        episode_step_max=episode_step_max,
    )
    # 如果不可视化节点就不用取返回值graph
    # 加入tensorboard
    writer = SummaryWriter(log_dir="logs/train")
    env.build_edge()

    # 加载之前的
    obs_states, edge_index, reward, dones, _ = env.get_obs()

    hetero_data = HeteroData()
    # 节点信息
    for key, _value in obs_states.items():
        hetero_data[key].x = _value
        # 边信息
    for key, _value in edge_index.items():
        node1, node2 = key.split("2")
        hetero_data[(f"{node1}", f"{key}", f"{node2}")].edge_index = _value
    agent = Agent(
        batch_size=batch_size,
        n_epochs=n_epochs,
        init_data=hetero_data,
        center_per_process=env.center_per_process,
        center_num=env.total_center_num,
        process_num=env.process_num,
        product_num=env.product_num,
    )

    agent.load_model("last_model.pth")
    memory = PPOMemory(
        batch_size,
        device,
    )

    logger.info(f"模型加载完成，环境初始化完成")
    now_time = datetime.now()

    # 添加计算图
    a, b = agent.network(obs_states, edge_index)

    writer.add_graph(
        agent.network,
        input_to_model=[obs_states, edge_index],
        verbose=False,
        use_strict_trace=False,
    )
    # 添加
    # show()
    episode = 0
    while total_step < init_step + max_steps:
        # logger.debug(f"第{total_step}步")
        total_step += 1
        agent.network.eval()
        with torch.no_grad():
            centers_power_action, center_func_action, centers_ratio, log_prob_power, log_prob_func = agent.get_action(
                obs_states, edge_index)
            value = agent.get_value(obs_states, edge_index)
        # 这个raw因为是字典，这里变了之后会影响get action中的raw
        # 后来还是改为了直接的tensor
        # for key, _value in raw.items():
        #    raw[key] = _value.cpu()
        # assert isinstance(raw, dict), "raw 不是字典"
        # assert raw.device != "cpu", "raw 不在cpu中"
        # logger.info(_raw)

        env.update(centers_power_action.cpu(), center_func_action.cpu(), centers_ratio.cpu())
        # 可视化状态
        # logger.debug(f"{total_step} {env.read_state()}")
        for i, mm in enumerate(env.read_state()['storage']):
            writer.add_scalar(f"step/state/{i}storage", mm, total_step)
        # 所以需要搬回cuda中
        # for key, _value in raw.items():
        #    raw[key] = _value.to(device)
        obs_states, edge_index, reward, dones, episode_step = env.get_obs()
        writer.add_scalar("step/reward", reward, total_step)
        memory.remember(obs_states,
                        edge_index,
                        value,
                        reward,
                        dones,
                        centers_power_action.cuda(),
                        center_func_action.cuda(),
                        centers_ratio,
                        log_prob_power,
                        log_prob_func)
        # 如果记忆数量等于batch_size就学习
        if memory.count == batch_size:
            learn_num += 1
            agent.network.train()
            loss = agent.learn(
                memory,
                last_node_state=obs_states,
                last_done=dones,
                edge_index=edge_index,
                mini_batch_size=batch_size // 2,
            )
            learn_time = (datetime.now() - now_time).seconds
            logger.info(f"第{learn_num}次学习，学习用时：{learn_time}秒")
            agent.save_model("last_model.pth")
            now_time = datetime.now()
            writer.add_scalar("loss", loss, total_step)
        if dones == 1:
            episode += 1
            logger.info(f"总步数：{total_step}，本次循环步数为：{episode_step}，奖励为{reward:.3f}")
            writer.add_scalar("reward", reward, episode)
            # with open("./log.csv", "a", newline="") as csvfile:
            #     csv_writer = csv.writer(csvfile)
            #     csv_writer.writerow([total_step, f"{reward:.3f}"])
            env.reset()
            obs_states, edge_index, _, _, _ = env.get_obs()
        if total_step % 500 == 0:
            agent.save_model("model_" + str(total_step) + ".pth")

    # # 神经网络要输出每个工作站的工作，功能和传输与否
    # agent.save_model("last_model.pth")
    # total_time = (datetime.now() - init_time).seconds // 60
    # with open("./log.csv", "a", newline="") as csvfile:
    #     csv_writer = csv.writer(csvfile)
    #     csv_writer.writerow([total_step, f"{reward:.3f}"])
    # print(f"总计用时：{total_time}分钟，运行{total_step}步，学习{learn_num}次")


if __name__ == '__main__':
    # logger.remove()
    # logger.add(sys.stderr, level='WARNING')
    main(512)
    # cProfile.run('main()', sort='cumulative')
