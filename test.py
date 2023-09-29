import numpy as np
import networkx as nx
from matplotlib import pyplot as plt
import torch
from GNNAgent import Agent, PPOMemory
from envClass import EnvRun
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

if __name__ == "__main__":
    function_num = 12
    work_cell_num = 100
    batch_size = 64

    total_step = 0
    max_steps = 500
    episode_step_max = 100
    product_goal = 500
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    env = EnvRun(
        work_cell_num=work_cell_num,
        function_num=function_num,
        device=device,
        episode_step_max=episode_step_max,
        product_goal=product_goal,
    )
    agent = Agent(
        work_cell_num,
        function_num,
        batch_size=batch_size,
        n_epochs=32,
    )
    graph = env.build_edge()
    writer = SummaryWriter(log_dir="logs/test")
    agent.load_model("last_model.pth")
    obs_states, edge_index, reward, dones, _ = env.get_obs()
    init_time = datetime.now()
    print(f"模型加载完成，环境初始化完成，当前时间{init_time.strftime('%Y-%m-%d %H:%M:%S')}")
    now_time = datetime.now()

    # 添加
    while total_step < max_steps:
        total_step += 1
        agent.network.eval()
        with torch.no_grad():
            raw, log_prob = agent.get_action(obs_states, edge_index)
            value = agent.get_value(obs_states, edge_index)
        # 这个raw少了
        env.update_all(raw.cpu())
        obs_states, edge_index, reward, dones, episode_step = env.get_obs()
        writer.add_scalars(
            "step/products",
            {
                f"产品{i}": env.total_products[i]
                for i in range(1, env.total_products.shape[0])
            },
            total_step,
        )
        writer.add_scalar("step/reward", reward, total_step)
        # 如果记忆数量等于batch_size就学习
        if dones == 1:
            print("=================")
            print(f"总步数：{total_step}，本次循环步数为：{episode_step}，奖励为{ reward:.3f}")
            writer.add_scalar("reward", reward, total_step)
            env.reset()
