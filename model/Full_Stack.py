from datetime import datetime

from PPOMemory import PPOMemory
from envClass import EnvRun
from loguru import logger
from algorithm.Genetic import GeneticAlgorithmNUMPY
import torch
from model.GNNAgent import Agent
from torch_geometric.data import HeteroData


def main():
    process_speed = [[5, 8, 3],
                     [10, 12, 6],
                     [15, 18, float('nan')],
                     [20, float('nan'), 10],
                     [12, 12, 8]]
    orders = [[100, 200, 400], [60, 50, 50], [200, 100, 105]]
    rmt_units = [16, 10, 10, 15, 10]
    # 自然选择部分
    pop_num = 100
    generation = 50
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(3407)
    batch_size = 32
    ga = GeneticAlgorithmNUMPY(pop_num, generation, orders[0], process_speed, rmt_units)
    best_time, best_solution = ga.evolve()
    total_step = 0
    init_step = 0
    max_steps = 512
    n_epochs = 12
    learn_num = 0
    env = EnvRun(order=orders[0], work_center_init_func=best_solution, speed_list=process_speed, device=device,
                 expected_step=best_time)
    env.build_edge()

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
    memory = PPOMemory(
        batch_size,
        device,
    )
    logger.info(f"模型加载完成，环境初始化完成")
    now_time = datetime.now()
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

        env.update(centers_power_action.cpu(), center_func_action.cpu(), centers_ratio.cpu(), total_step)
        # 可视化状态
        # logger.debug(f"{total_step} {env.read_state()}")
        # p = env.read_state()
        # a = {f"{i}storage": mm[0] for i, mm in enumerate(p['storage'])}
        # writer.add_scalars(f"storage", a, total_step)

        # for i, (mm, _id) in enumerate(p['storage'][-3:]):
        #     writer.add_scalar(f"storage/storage_{i}_product{_id}", mm, total_step)
        #
        # for i, n in enumerate(p['center']):
        #     q = {"func": n[0], 'status': n[1], 'material': n[2], 'product': n[2]}
        #     writer.add_scalars(f"centers/center_{i}", q, total_step)
        # writer.add_scalars(f"storage/total", p['total_storage_num'], total_step)

        # 所以需要搬回cuda中
        # for key, _value in raw.items():
        #    raw[key] = _value.to(device)
        obs_states, edge_index, reward, dones, episode_step = env.get_obs()
        # writer.add_scalar("step/reward", reward, total_step)
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
        if dones == 1:
            episode += 1
            logger.info(f"总步数：{total_step}，本次循环步数为：{episode_step}，奖励为{reward:.3f}")
            env.reset()
            obs_states, edge_index, _, _, _ = env.get_obs()
        if total_step % 500 == 0:
            agent.save_model("model_" + str(total_step) + ".pth")
if __name__ == '__main__':
    main()