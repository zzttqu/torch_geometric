import gc
from datetime import datetime

import numpy as np
import torch

from loguru import logger
from torch_geometric.data import HeteroData

from model.PPOMemory import PPOMemory
from model.algorithm.Genetic import GeneticAlgorithmNUMPY
from model.GNNAgent import Agent
from model.envClass import EnvRun
from model.utils.DataUtils import data_generator


class Train:
    def __init__(self, data_len=1, process_num=5, product_num=5):
        torch.manual_seed(3407)
        np.random.seed(3407)
        self.speed_list, self.order_list, self.rmt_units_num_list = data_generator(process_num, product_num, data_len)
        # 自然选择部分
        self.pop_num = 100
        self.generation = 50
        self.batch_size = 32
        # self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.data_len = data_len
        self.device = torch.device('cpu')

    def init_setting(self, env_index: int, first_init: bool):
        """
        初始化环境为指定索引值
        Args:
            first_init:
            env_index:

        Returns:

        """
        # 初始化模型用
        ga = GeneticAlgorithmNUMPY(self.pop_num, self.generation,
                                   self.order_list[env_index],
                                   self.speed_list[env_index],
                                   self.rmt_units_num_list[env_index])
        self.best_time, best_solution = ga.evolve()
        self.env = EnvRun(device=self.device)

        self.env.reinit(order=self.order_list[env_index], work_center_init_func=best_solution,
                        speed_list=self.speed_list[env_index],
                        expected_step=self.best_time, episode_step_max=self.best_time * 2)

        self.obs_states, self.edge_index, _, _, _, _ = self.env.get_obs()
        self.epsoide_max_steps = min(self.batch_size * 20, self.best_time * 10)
        n_epochs = 8
        hetero_data = HeteroData()
        # 节点信息
        for key, _value in self.obs_states.items():
            hetero_data[key].x = _value
            # 边信息
        for key, _value in self.edge_index.items():
            node1, node2 = key.split("2")
            hetero_data[(f"{node1}", f"{key}", f"{node2}")].edge_index = _value
        self.agent = Agent(
            batch_size=self.batch_size,
            n_epochs=n_epochs,
            init_data=hetero_data,
            device=self.device
        )
        load_model_name = "best2.pth"
        logger.debug(f'加载了模型{load_model_name}')
        self.agent.load_model(load_model_name)

        self.memory = PPOMemory(
            self.batch_size,
            self.device,
        )
        self.agent.init(self.batch_size, self.env.center_per_process, self.env.total_center_num)

        self.total_step = 0
        self.episode = 0
        self.learn_num = 0
        # 返回当前env的基本参数
        # self.order_list[env_index]
        # self.speed_list[env_index]
        # self.rmt_units_num_list[env_index]
        return {'order': self.order_list[env_index].tolist(),
                'speed': self.speed_list[env_index].tolist(),
                'rmt_units_num': self.rmt_units_num_list[env_index].tolist(), 'GA_best_solution': self.best_time}

    def step(self):
        self.agent.network.eval()
        read_state = self.env.read_state()
        self.total_step += 1
        with torch.no_grad():
            centers_power_action, center_func_action, centers_ratio, log_prob_power, log_prob_func, _ = self.agent.get_action(
                self.obs_states, self.edge_index)
            value = self.agent.get_value(self.obs_states, self.edge_index)

        self.env.update(centers_power_action.cpu(), center_func_action.cpu(), centers_ratio.cpu())
        self.obs_states, self.edge_index, reward, dones, episode_step, finish_state = self.env.get_obs()
        self.memory.remember(self.obs_states,
                             self.edge_index,
                             value,
                             reward,
                             dones,
                             centers_power_action.to(self.device),
                             center_func_action.to(self.device),
                             centers_ratio,
                             log_prob_power,
                             log_prob_func)
        # 如果记忆数量等于batch_size就学习
        if self.memory.count == self.batch_size:
            self.learn_num += 1
            self.agent.network.train()
            loss = self.agent.learn(
                self.memory,
                last_node_state=self.obs_states,
                last_done=dones,
                edge_index=self.edge_index,
                mini_batch_size=self.batch_size // 2,
                progress=episode_step / self.epsoide_max_steps,
            )
            self.agent.save_model("last_model.pth")
        if dones == 1:
            self.episode += 1
            logger.info(
                f"总步数：{self.total_step}，本次循环步数为：{episode_step}，奖励为{reward:.3f}，订单完成状态为：{finish_state.tolist()}")
            self.env.reset()
            self.obs_states, self.edge_index, _, _, _, _ = self.env.get_obs()
        if self.total_step % 500 == 0:
            self.agent.save_model("model_" + str(self.total_step) + ".pth")
        # 可视化状态
        return {"step": self.total_step, "state": read_state}

    def test(self):
        self.agent.network.eval()
        read_state = self.env.read_state()
        with torch.no_grad():
            centers_power_action, center_func_action, centers_ratio, log_prob_power, log_prob_func, _ = self.agent.get_action(
                self.obs_states, self.edge_index)
        self.env.update(centers_power_action.cpu(), center_func_action.cpu(), centers_ratio.cpu())
        self.obs_states, self.edge_index, reward, dones, episode_step, finish_state = self.env.get_obs()
        if dones == 1:
            logger.info(
                f"本次循环步数为：{episode_step}，奖励为{reward:.3f}，订单完成状态为：{finish_state.tolist()}")
            self.env.reset()
            self.obs_states, self.edge_index, _, _, _, _ = self.env.get_obs()
        # 可视化状态
        return {"step": episode_step - 1, "dones": dones, "state": read_state}

    def reset(self):
        self.total_step = 0
        self.episode = 0
        self.learn_num = 0
        self.env.reset()
        self.obs_states, self.edge_index, _, _, _, _ = self.env.get_obs()

    def delete(self):
        torch.cuda.empty_cache()
        del self
        gc.collect()
        return


if __name__ == '__main__':
    train = Train(1, 3, 5)
    logger.info(train.init_setting(0, True))

    for _ in range(10):
        res = train.step()
