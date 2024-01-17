"""
本文件主要用于测试包括遗传算法和机器学习
"""
from datetime import datetime

from torch.utils.tensorboard import SummaryWriter

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
    orders = [[100, 200, 400], [600, 500, 500], [200, 100, 105], [180, 300, 200]]
    rmt_units = [16, 10, 10, 15, 10]
    # 自然选择部分
    pop_num = 100
    generation = 50
    torch.manual_seed(3407)
    batch_size = 32
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')

    # 初始化模型用
    ga = GeneticAlgorithmNUMPY(pop_num, generation, orders[0], process_speed, rmt_units)
    best_time, best_solution = ga.evolve()
    env = EnvRun(device=device)
    env.initialize(order=orders[0], work_center_init_func=best_solution, speed_list=process_speed,
                   expected_step=best_time, episode_step_max=best_time * 2)
    obs_states, edge_index, _, _, _, _ = env.get_obs()
    n_epochs = 8
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
        device=device
    )
    # load_model_name = "last_model.pth"
    # logger.debug(f'加载了模型{load_model_name}')
    # agent.load_model(load_model_name)
    # 添加计算图
    a, b = agent.network(obs_states, edge_index)
    writer = SummaryWriter(log_dir="logs/train")
    writer.add_graph(
        agent.network,
        input_to_model=[obs_states, edge_index],
        verbose=False,
        use_strict_trace=False,
    )
    del obs_states, edge_index
    torch.cuda.empty_cache()

    total_step = 0
    episode = 0
    indexes = torch.randint(0, len(orders), size=(3,))

    for index in indexes:
        # TODO 应该先测试一下当前模型的效果
        order = orders[index]
        torch.cuda.empty_cache()
        logger.info(f'=======当前订单为: {order}=======')
        ga = GeneticAlgorithmNUMPY(pop_num, generation, order, process_speed, rmt_units)
        best_time, best_solution = ga.evolve()
        logger.success(f'算法最优{best_time},配置为{best_solution}')
        init_step = total_step
        init_episode = episode
        batch_size = 16
        max_steps = batch_size * 20

        learn_num = 0

        env.reinit(order=order, work_center_init_func=best_solution, speed_list=process_speed,
                   expected_step=best_time, episode_step_max=best_time * 2)
        obs_states, edge_index, reward, dones, _, _ = env.get_obs()
        agent.init(batch_size, env.center_per_process, env.process_num, env.total_center_num)
        memory = PPOMemory(
            batch_size,
            device,
        )
        logger.info(f"环境初始化完成,训练步数: {max_steps}，批次大小: {batch_size}")
        now_time = datetime.now()
        episode = init_episode
        while total_step < init_step + max_steps:
            # logger.debug(f"第{total_step}步")
            total_step += 1
            agent.network.eval()
            with torch.no_grad():
                centers_power_action, center_func_action, centers_ratio, log_prob_power, log_prob_func, _ = agent.get_action(
                    obs_states, edge_index)
                value = agent.get_value(obs_states, edge_index)

            env.update(centers_power_action.cpu(), center_func_action.cpu(), centers_ratio.cpu(), total_step)
            # 可视化状态
            # logger.debug(f"{total_step} {env.read_state()}")
            p = env.read_state()
            a = {f"{i}storage": mm[0] for i, mm in enumerate(p['storage'][-3:])}
            b = {f"{i}storage": mm[0] for i, mm in enumerate(p['storage'][:3])}
            writer.add_scalars(f"storage/product", a, total_step)
            writer.add_scalars(f"storage/material", b, total_step)

            # for i, (mm, _id) in enumerate(p['storage'][-3:]):
            #     writer.add_scalar(f"storage/storage_{i}_product{_id}", mm, total_step)
            #
            # for i, n in enumerate(p['center']):
            #     q = {"func": n[0], 'status': n[1], 'material': n[2], 'product': n[2]}
            #     writer.add_scalars(f"centers/center_{i}", q, total_step)
            writer.add_scalars(f"storage/total", p['total_storage_num'], total_step)

            obs_states, edge_index, reward, dones, episode_step, finish_state = env.get_obs()
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
                    progress=episode_step / max_steps,
                )
                # learn_time = (datetime.now() - now_time).seconds
                # logger.info(f"第{learn_num}次学习，学习用时：{learn_time}秒")
                # now_time = datetime.now()
                agent.save_model("last_model.pth")
            if dones == 1:
                episode += 1
                logger.info(
                    f"总步数：{total_step}，本次循环步数为：{episode_step}，奖励为{reward:.3f}，订单完成状态为：{finish_state.tolist()}")
                writer.add_scalars("episode", {'reward': reward, 'time': episode_step}, episode)

                env.reset()
                obs_states, edge_index, _, _, _, _ = env.get_obs()
            if total_step % 500 == 0:
                agent.save_model("model_" + str(total_step) + ".pth")


if __name__ == '__main__':
    main()
