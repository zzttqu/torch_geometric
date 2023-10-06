from datetime import datetime
import os
from typing import Dict, List, Tuple
from torch.distributions import Categorical
from torch.utils.data import BatchSampler, SubsetRandomSampler
from torch_geometric.data import Data, Batch, HeteroData
from torch_geometric.loader import DataLoader
from GNNNet import GNNNet, HGTNet
import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from PPOMemory import PPOMemory

# T.Constant()


class Agent:
    def __init__(
        self,
        work_cell_num,
        center_num,
        batch_size,
        n_epochs,
        init_data: HeteroData,
        clip=0.2,
        lr=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
    ):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.work_cell_num = work_cell_num
        self.center_num = center_num
        self.gamma = gamma
        self.batch_size = batch_size
        self.gae_lambda = gae_lambda
        self.n_epochs = n_epochs

        self.clip = clip

        self.metadata = init_data.metadata()
        self.undirect_data = init_data
        # 如果为异质图
        assert init_data is not None, "init_data是异质图必需的"
        self.update_heterodata(init_data)
        self.network = HGTNet(
            2,
            self.undirect_data,
        ).to(self.device)
        # 使用to_hetro相当于变了一个模型，还得todevice
        """self.network = to_hetero(self.network, self.metadata, aggr="sum").to(
            self.device
        )"""
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=lr, eps=1e-5)

    def update_heterodata(self, data: HeteroData):
        """更新heterodata"""
        # 去掉无向边
        # self.undirect_data = T.ToUndirected()(data)
        # metadata函数第一个返回值是节点列表，第二个返回值是边列表
        self.metadata = self.undirect_data.metadata()

    def get_value(
        self,
        state: Dict[str, torch.Tensor],
        edge_index: Dict[Tuple[str], torch.Tensor],
    ) -> torch.Tensor:
        _, value = self.network(state, edge_index)
        return value

    def get_action(
        self,
        state: Dict[str, torch.Tensor],
        edge_index: Dict[Tuple[str], torch.Tensor],
        all_action: torch.Tensor = None,
    ) -> (Dict[str, torch.Tensor], Dict[str, torch.Tensor]):
        # hetero_data = T.ToUndirected()(hetero_data)

        logits, _ = self.network(state, edge_index)
        # 切割出workcell
        all_logits = logits

        # 第一项是功能动作，第二项是是否接受上一级运输
        # 目前不需要对center进行动作，所以存储后可以不用
        all_logits = all_logits.view((-1, 2))
        all_dist = Categorical(logits=all_logits)
        # 判断action是否为空
        if all_action is None:
            all_action = all_dist.sample()
        # 只管采样，不管是哪类节点
        # 前边是workcell，后边是center
        log_probs = torch.stack([all_dist.log_prob(all_action)]).sum(0)

        return all_action, log_probs

    def get_batch_values(
        self,
        node: List[Dict[str, torch.Tensor]],
        edge: List[Dict[Tuple[str], torch.Tensor]],
        mini_batch_size,
    ):
        all_values = []
        for i in range(mini_batch_size):
            value = self.get_value(node[i], edge[i])
            all_values.append(value)
        all_values = torch.stack(all_values)
        return all_values

    def get_batch_actions_probs(
        self,
        mini_batch_size,
        node: List[Dict[str, torch.Tensor]],
        edge: List[Dict[Tuple[str], torch.Tensor]],
        action_list: List[torch.Tensor],
    ):
        all_log_probs = []
        for i in range(mini_batch_size):
            action, log_probs = self.get_action(node[i], edge[i], action_list[i])
            all_log_probs.append(log_probs)
        log_probs_list = torch.cat(all_log_probs, dim=0).view((mini_batch_size, -1))
        return log_probs_list.sum(0)

    def load_model(self, name):
        # 如果没有文件需要跳过
        name = f"./model/{name}"
        if not os.path.exists(name):
            return
        self.network.load_model(name)

    def save_model(self, name):
        name = f"./model/{name}"
        self.network.save_model(name)

    def learn(
        self,
        ppo_memory: PPOMemory,
        last_node_state,
        last_done,
        edge_index,
        mini_batch_size=16,
    ):
        (
            nodes,
            edges,
            values,
            rewards,
            dones,
            actions,
            log_probs,
        ) = ppo_memory.generate_batches()
        # flat_states = batches.reshape(-1, batches.shape[-1])
        # flat_actions = total_actions.view(-1, total_actions.shape[-1])
        # flat_probs = log_probs.view(-1, log_probs.shape[-1])
        # 计算GAE
        with torch.no_grad():
            last_value = self.get_value(last_node_state, edge_index)
            advantages = torch.zeros_like(rewards).to(self.device)
            for t in reversed(range(self.batch_size)):
                last_gae_lam = 0
                if t == self.batch_size - 1:
                    last_gae_lam = 0
                    if t == self.batch_size - 1:
                        next_nonterminal = 1.0 - last_done
                        next_values = last_value
                    else:
                        next_nonterminal = 1.0 - dones[t + 1]
                        next_values = values[t + 1]
                    delta = (
                        rewards[t]
                        + self.gamma * next_values * next_nonterminal
                        - values[t]
                    )
                    last_gae_lam = (
                        delta
                        + self.gamma * self.gae_lambda * next_nonterminal * last_gae_lam
                    )
                    advantages[t] = last_gae_lam
            # 规范化
            # advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)
            # 这个是value的期望值
            returns: torch.Tensor = advantages + values
        flat_advantages = advantages.view(-1)
        flat_returns = returns.view(-1)
        total_loss = torch.tensor(0.0)
        for _ in range(self.n_epochs):
            # 这里是设置minibatch，也就是送入图神经网络的大小
            for index in BatchSampler(
                SubsetRandomSampler(range(self.batch_size)), mini_batch_size, False
            ):
                # batch_time = datetime.now()
                mini_nodes = [nodes[i] for i in index]
                mini_edges = [edges[i] for i in index]
                mini_actions = [actions[i] for i in index]

                # 这里必须用stack因为是堆叠出新的维度，logprobs只有一个维度
                mini_probs = torch.stack([log_probs[i] for i in index])
                new_log_prob = self.get_batch_actions_probs(
                    mini_batch_size,
                    mini_nodes,
                    mini_edges,
                    mini_actions,
                )

                # TODO 这里感觉应该连接起来变成一个

                ratios = torch.exp(new_log_prob - mini_probs).sum(1)
                surr1 = ratios * flat_advantages[index]
                surr2 = torch.clamp(ratios, 1 - self.clip, 1 + self.clip)
                actor_loss: torch.Tensor = -torch.min(surr1, surr2)
                new_value = self.get_batch_values(
                    mini_nodes, mini_edges, mini_batch_size
                )
                critic_loss = F.mse_loss(flat_returns[index], new_value.view(-1))
                total_loss: torch.Tensor = actor_loss.mean() + 0.5 * critic_loss
                self.optimizer.zero_grad()
                total_loss.backward()
                # 裁减
                # torch.nn.utils.clip_grad_norm_(self.network.parameters(), 10)
                self.optimizer.step()
                # batch_use_time = (datetime.now() - batch_time).microseconds

        return total_loss
