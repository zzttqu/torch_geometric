import os
from typing import Dict, List, Tuple

from loguru import logger
import torch
import torch.nn.functional as F
from torch import Tensor

from model.GNNNet import HGTNet
from model.PPOMemory import PPOMemory
from torch.distributions import Categorical
from torch.utils.data import BatchSampler, SequentialSampler
from torch_geometric.data import HeteroData


class Agent:
    def __init__(
            self,
            batch_size,
            n_epochs,
            init_data: HeteroData,
            center_per_process,
            work_center2cell_list,
            per_process_num,
            clip=0.2,
            lr=3e-4,
            gamma=0.99,
            gae_lambda=0.95,
    ):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.center_per_process = center_per_process
        self.per_process_num = per_process_num
        self.work_center2cell_list = work_center2cell_list

        self.gamma = gamma
        self.batch_size = batch_size
        self.gae_lambda = gae_lambda
        self.n_epochs = n_epochs
        self.clip = clip
        # 如果为异质图
        assert init_data is not None, "init_data是异质图必需的"
        self.network = HGTNet(
            init_data,
        ).to(self.device)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=lr, eps=1e-5)

    def get_value(
            self,
            state: Dict[str, torch.Tensor],
            edge_index: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        _, value = self.network(state, edge_index)
        return value

    def get_action(
            self,
            state: Dict[str, torch.Tensor],
            edge_index: Dict[str, torch.Tensor],
            all_action: Dict[str, torch.Tensor] = None,  # type: ignore
    ) -> tuple[Tensor, Tensor, Tensor]:
        # hetero_data = T.ToUndirected()(hetero_data)
        all_logits, _ = self.network(state, edge_index)
        total_center_num = len(all_logits["center"])
        cell_logits, center_logits = all_logits.values()
        log_probs = torch.zeros((total_center_num, 2), dtype=torch.float32)
        actions = torch.zeros((total_center_num, 2), dtype=torch.float32)
        center_ratio = torch.zeros(total_center_num, dtype=torch.float32)
        # self.work_center_process中记录了各个process的workcenter数量
        center_id = 0
        # 首先生产
        for process, center_num in enumerate(self.center_per_process):
            assert isinstance(center_num, Tensor), "center_num 必须是 Tensor"
            # 为了保证不影响softmax，如果是0会导致最后概率不为0，根据每个工序包括的功能数量初始化
            _ratio = torch.full((self.per_process_num[process], center_num), -torch.inf)
            for process_index, cell_ids in enumerate(self.work_center2cell_list[center_id:center_id + center_num]):
                cell_slice = cell_logits[cell_ids]
                center_slice = center_logits[center_id, :]
                center_dist = Categorical(logits=center_slice)
                cell_dist = Categorical(logits=cell_slice[:, 0])
                # 这里其实不影响，因为实际上是修改的workcell，但是如果这个工作中心没有这个功能，其实工作单元也没有，那么这个输出的index其实就是celllist的index
                activate_func = cell_dist.sample()
                log_probs[center_id, 0] = cell_dist.log_prob(activate_func)
                # 这里是放置当前工序各个product的分配率
                _ratio[activate_func, process_index] = cell_slice[activate_func, 1]
                on_or_off = center_dist.sample()
                log_probs[center_id, 1] = center_dist.log_prob(on_or_off)
                actions[center_id, 0] = activate_func
                actions[center_id, 1] = on_or_off
            # 还需要考虑如果所有的cell都选了一个func就会导致全是inf，softmax后的tensor为nan
            _ratio = torch.softmax(_ratio, dim=1)
            # 如果有nan就改成0
            _ratio_without_nan = torch.where(torch.isnan(_ratio), torch.tensor(0.0), _ratio)
            # 因为center不能同时出现在两个工序中，所以_ratio必定同一列只有一个非0的，所以累加起来就变成center_num长度的数组了，就可以给center赋值了，
            # 然后交给之后的物料转运
            ratios = torch.sum(_ratio_without_nan, dim=0)
            center_ratio[center_id:center_id + center_num] = ratios
            # 更新center的id，方便循环内部使用
            center_id += center_num

        """assert isinstance(all_logits, dict), "必须是dict类型"
        # 这里是GNN的输出，是每个节点一个
        # 第一项是功能动作，第二项是是否接受上一级运输
        # 目前不需要对center进行动作，所以存储后可以不用

        all_dist: Dict[str, Categorical] = {}
        for key, logit in all_logits.items():
            all_dist[key] = Categorical(logits=logit)
        # 判断action是否为空
        log_probs = torch.zeros(0, dtype=torch.float32, device=self.device)
        # if all_action is not None:
        #     logger.info(all_action)
        if all_action is None:
            all_action = {}
            for key, dist in all_dist.items():
                all_action[key] = dist.sample()
        for key, _dist in all_dist.items():
            tmp = torch.stack([_dist.log_prob(all_action[key])]).sum(0)
            log_probs = torch.cat((log_probs, tmp), dim=0)
        # logger.debug(log_probs)
        # 只管采样，不管是哪类节点
        # 前边是workcell，后边是center
        # log_probs = torch.stack([all_dist.log_prob(all_action)]).sum(0)"""

        return actions, center_ratio, log_probs

    def get_batch_values(
            self,
            node: List[Dict[str, torch.Tensor]],
            edge: List[Dict[str, torch.Tensor]],
            mini_batch_size: int,
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
            edge: List[Dict[str, torch.Tensor]],
            action_list: List[Dict[str, torch.Tensor]],
    ):
        all_log_probs = []
        for i in range(mini_batch_size):
            _, log_probs = self.get_action(node[i], edge[i], action_list[i])
            # 这个probs本来就是一维的，其实不用cat，stack更好吧
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
            mini_batch_size,
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
                    SequentialSampler(range(self.batch_size)), mini_batch_size, False
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
                # logger.debug(f"new_log_prob: {new_log_prob}")

                # 这里感觉应该连接起来变成一个

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
