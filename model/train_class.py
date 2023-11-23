from datetime import datetime

import torch

from loguru import logger
from torch_geometric.data import HeteroData

from model.GNNAgent import Agent
from model.PPOMemory import PPOMemory
from model.envClass import EnvRun


class Train:
    """
    Train类就是用来做训练的，环境是不能变的，只要开始训练就在这个类中，训练参数必须一开始就制定好
    """

    def __init__(self,
                 function_num,
                 work_center_num,
                 func_per_center=2,
                 max_steps=0,
                 episode_step_max=32, n_epochs=8,
                 batch_size=32,
                 tensorboard_log=False, load_model=True):

        self.learn_num = 0
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # 神奇trick
        torch.manual_seed(3407)
        self.function_num = function_num
        self.work_center_num = work_center_num
        self.tensorboard_log = tensorboard_log
        self.load_model = load_model
        self.batch_size = batch_size
        self.episode_step_max = episode_step_max
        self.n_epochs = n_epochs
        if max_steps == 0:
            max_steps = self.batch_size * 10
        self.max_steps = max_steps
        self.total_step = 0
        self.init_step = 0
        self.init_time = datetime.now()

        # 初始化环境
        self.env = EnvRun(
            work_center_num=work_center_num,
            fun_per_center=func_per_center,
            function_num=function_num,
            device=self.device,
            episode_step_max=episode_step_max,
            product_goal_scale=0.2,
        )
        # 初始化metadata
        obs_states, edge_index, reward, dones, _ = self.env.get_obs()
        logger.info(f"加工能力为{self.env.product_capacity}")
        # 构建metaData
        self.hetero_data = HeteroData()
        # 节点信息
        for key, _value in obs_states.items():
            self.hetero_data[key].x = _value
            # 边信息
        for key, _value in edge_index.items():
            node1, node2 = key.split("_to_")
            self.hetero_data[(f"{node1}", f"{key}", f"{node2}")].edge_index = _value
        # 初始化memory
        self.memory = PPOMemory(
            batch_size,
            self.device,
        )
        self.agent = Agent(
            batch_size=batch_size,
            n_epochs=self.n_epochs,
            init_data=self.hetero_data,
        )

        if self.tensorboard_log:
            from torch.utils.tensorboard.writer import SummaryWriter
            self.writer = SummaryWriter(log_dir="logs/train")
            obs_states, edge_index, reward, dones, _ = self.env.get_obs()
            # 添加计算图
            _, _ = self.agent.network(obs_states, edge_index)

            self.writer.add_graph(
                self.agent.network,
                input_to_model=[obs_states, edge_index],
                verbose=False,
                use_strict_trace=False,
            )

    def train_local(self):
        # 初始化memory

        if self.load_model:
            self.agent.load_model("last_model.pth")
        now_time = datetime.now()

        # 只要没step，都是初始状态的obs
        obs_states, edge_index, reward, dones, _ = self.env.get_obs()
        while self.total_step < self.init_step + self.max_steps:
            self.total_step += 1
            self.agent.network.eval()
            with torch.no_grad():
                # raw是一个2*节点数量
                raw, log_prob = self.agent.get_action(obs_states, edge_index)
                value = self.agent.get_value(obs_states, edge_index)
            if self.tensorboard_log:
                self.writer.add_scalars(
                    "step/products",
                    {
                        f"产品{i}": self.env.storage_list[i].get_product_num()
                        for i in range(0, len(self.env.storage_list))
                    },
                    self.total_step,
                )
            # 这个raw因为是字典，这里变了之后会影响get action中的raw
            # 后来还是改为了直接的tensor
            # for key, _value in raw.items():
            #    raw[key] = _value.cpu()
            assert isinstance(raw, dict), "raw 不是字典"
            # assert raw.device != "cpu", "raw 不在cpu中"
            _raw = {}
            for key, _value in raw.items():
                _raw[key] = _value.cpu()
            self.env.update_all(_raw)
            obs_states, edge_index, reward, dones, episode_step = self.env.get_obs()

            if self.tensorboard_log:
                self.writer.add_scalar("step/reward", reward, self.total_step)
            self.memory.remember(obs_states, edge_index, value, reward, dones, raw, log_prob)
            # 如果记忆数量等于batch_size就学习
            if self.memory.count == self.batch_size:
                self.learn_num += 1
                self.agent.network.train()
                loss = self.agent.learn(
                    self.memory,
                    last_node_state=obs_states,
                    last_done=dones,
                    edge_index=edge_index,
                    mini_batch_size=self.batch_size // 2,
                )
                learn_time = (datetime.now() - now_time).seconds
                print(f"第{self.learn_num}次学习，学习用时：{learn_time}秒")
                self.agent.save_model("last_model.pth")
                now_time = datetime.now()
                if self.tensorboard_log:
                    self.writer.add_scalar("loss", loss, self.total_step)
            if dones == 1:
                print("=================")
                print(f"总步数：{self.total_step}，本次循环步数为：{episode_step}，奖励为{reward:.3f}")
                if self.tensorboard_log:
                    self.writer.add_scalar("reward", reward, self.total_step)
                self.env.reset()
                obs_states, edge_index, _, _, _ = self.env.get_obs()
            if self.total_step % 500 == 0:
                self.agent.save_model("model_" + str(self.total_step) + ".pth")

        # 神经网络要输出每个工作站的工作，功能和传输与否
        self.agent.save_model("last_model.pth")
        # 清理缓存，卸载模型，保留环境
        torch.cuda.empty_cache()
        # del agent
        # del memory
        total_time = (datetime.now() - self.init_time).seconds // 60
        logger.info(f"总计用时：{total_time}分钟，运行{self.total_step}步，学习{self.learn_num}次")

    def train_online(self, step, stop=False):
        """
        区别就是每调用一次只走一步
        Args:
            step:
            stop:

        Returns:

        """
        # 初始化memory

        if self.load_model:
            self.agent.load_model("last_model.pth")
            # if save:
        #     self.agent.save_model("last_model.pth")
        #     yield

        now_time = datetime.now()

        # 只要没step，都是初始状态的obs
        obs_states, edge_index, reward, dones, _ = self.env.get_obs()
        # logger.info((obs_states, edge_index))
        for step in range(step):
            self.total_step += 1
            self.agent.network.eval()
            with torch.no_grad():
                # raw是一个2*节点数量
                raw, log_prob = self.agent.get_action(obs_states, edge_index)
                value = self.agent.get_value(obs_states, edge_index)
            if self.tensorboard_log:
                self.writer.add_scalars(
                    "step/products",
                    {
                        f"产品{i}": self.env.storage_list[i].get_product_num()
                        for i in range(0, len(self.env.storage_list))
                    },
                    self.total_step,
                )
            assert isinstance(raw, dict), "raw 不是字典"
            _raw = {}
            for key, _value in raw.items():
                _raw[key] = _value.cpu()
            self.env.update_all(_raw)
            obs_states, edge_index, reward, dones, episode_step = self.env.get_obs()

            if self.tensorboard_log:
                self.writer.add_scalar("step/reward", reward, self.total_step)
            self.memory.remember(obs_states, edge_index, value, reward, dones, raw, log_prob)
            # 如果记忆数量等于batch_size就学习
            if self.memory.count == self.batch_size:
                self.learn_num += 1
                self.agent.network.train()
                yield "training"
                loss = self.agent.learn(
                    self.memory,
                    last_node_state=obs_states,
                    last_done=dones,
                    edge_index=edge_index,
                    mini_batch_size=self.batch_size // 2,
                )
                learn_time = (datetime.now() - now_time).seconds
                logger.info(f"第{self.learn_num}次学习，学习用时：{learn_time}秒")
                now_time = datetime.now()
                # self.agent.save_model("last_model.pth")
                if self.tensorboard_log:
                    self.writer.add_scalar("loss", loss, self.total_step)
            if self.total_step % 500 == 0:
                self.agent.save_model("model_" + str(self.total_step) + ".pth")
            yield [self.total_step, self.env.online_state(), reward, dones]
            if dones == 1:
                logger.info(f"总步数：{self.total_step}，本次循环步数为：{episode_step}，奖励为{reward:.3f}")
                if self.tensorboard_log:
                    self.writer.add_scalar("reward", reward, self.total_step)
                self.env.reset()
                obs_states, edge_index, _, _, _ = self.env.get_obs()
        # 清理缓存，卸载模型，保留环境
        # torch.cuda.empty_cache()
        # total_time = (datetime.now() - self.init_time).seconds // 60
        # logger.info(f"总计用时：{total_time}分钟，运行{self.total_step}步，学习{self.learn_num}次")
        # return "finish"


# import os
#
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
if __name__ == '__main__':
    a = Train(2, 2, 64 * 10, load_model=False)
    a.train_local()
