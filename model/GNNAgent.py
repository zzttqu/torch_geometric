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
            process_num,
            center_num,
            product_num,
            clip=0.2,
            lr=3e-4,
            gamma=0.99,
            gae_lambda=0.95,
    ):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.center_per_process = center_per_process
        self.center_num = center_num
        self.process_num = process_num
        self.product_num = product_num

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
            centers_power_action: torch.Tensor = None,  # type: ignore
            center_func_action: torch.Tensor = None,  # type: ignore
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        all_logits, _ = self.network(state, edge_index)
        cell_logits, center_logits = all_logits.values()
        # self.work_center_process中记录了各个process的workcenter数量

        # 工作中心启动还是停止

        _center_ratio = torch.full((self.process_num, self.center_num), -torch.inf)
        # 启动哪个功能
        center_func_logits = cell_logits[:, 0].reshape(self.center_num, -1).clone()
        # 直接折叠成centernum的形状，就可以选择
        center_ratio_logits = cell_logits[:, 1].reshape(self.center_num, -1).clone()
        _center_func_dist = Categorical(logits=center_func_logits)
        if center_func_action is None:
            center_func_action = _center_func_dist.sample()
        log_probs_center_func = _center_func_dist.log_prob(center_func_action)
        centers_dist = Categorical(logits=center_logits)
        if centers_power_action is None:
            centers_power_action = centers_dist.sample()
        log_probs_center_power = centers_dist.log_prob(centers_power_action)
        # 物料分配概率
        # 先创建一个[0,1,2]的tensor，然后增加一个维度变成[[0,1,2]]，因为center_func_action是[1,2,1,2,0......]，
        # 所以也需要扩展把这个变成竖着的，然后两个求布尔，就和workcenter中生成边是一个道理
        mask = torch.eq(torch.arange(center_ratio_logits.size(1), device=self.device).unsqueeze(
            0), center_func_action.unsqueeze(1))
        # 只有既开启并有功能的cell才能参与分配,需要新建一个维度来广播
        # 逐个元素求与，需要同时满足
        mask = torch.logical_and(mask, centers_power_action.unsqueeze(1)).bool()
        # 未被选中的cell的分配率为负无穷，要取反，因为都满足才能留下，所以如果True被-inf就寄了，所以需要取反
        center_ratio_logits[~mask] = -torch.inf
        # logger.debug(center_ratio_logits)
        # 提取每个center被选中的func的分配率
        # TODO 或者改成分配顺序，因为只能按speed进行分配，这样反而还可以提高速度
        _center_id = 0
        for num in self.center_per_process:
            # 每道工序进行softmax
            center_ratio_logits[_center_id:_center_id + num] = F.softmax(
                center_ratio_logits[_center_id:_center_id + num], dim=0)
            _center_id = _center_id + num
        center_ratio_logits = torch.nan_to_num(center_ratio_logits, nan=0, posinf=0, neginf=0)
        centers_ratio: Tensor = center_ratio_logits.sum(dim=1)
        """# 首先生产
        for process, center_num in enumerate(self.center_per_process):
            assert isinstance(center_num, Tensor), "center_num 必须是 Tensor"
            # 为了保证不影响softmax，如果是0会导致最后概率不为0，根据每个工序包括的功能数量初始化
            _ratio = torch.full((self.per_process_num[process], center_num), -torch.inf)
            # TODO 急需优化，此处耗时过长
            for process_index, cell_ids in enumerate(self.work_center2cell_list[center_id:center_id + center_num]):
                cell_slice = cell_logits[cell_ids]
                cell_dist = Categorical(logits=cell_slice[:, 0])
                # 这里其实不影响，因为实际上是修改的workcell，但是如果这个工作中心没有这个功能，其实工作单元也没有，那么这个输出的index其实就是celllist的index
                activate_func = cell_dist.sample()
                log_probs[center_id + process_index, 0] = cell_dist.log_prob(
                    activate_func) if all_action is None else cell_dist.log_prob(
                    all_action[process_index + center_id, 0])
                # 这里是放置当前工序各个product的分配率
                _ratio[activate_func, process_index] = cell_slice[activate_func, 1]
                actions[center_id + process_index, 0] = activate_func

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

        assert isinstance(all_logits, dict), "必须是dict类型"
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

        return centers_power_action, center_func_action, centers_ratio, log_probs_center_power, log_probs_center_func

    def get_batch_values(
            self,
            node: List[Dict[str, torch.Tensor]],
            edge: List[Dict[str, torch.Tensor]],
            mini_batch_size: int,
    ):
        all_values = torch.zeros(mini_batch_size, device=self.device, dtype=torch.float)
        for i in range(mini_batch_size):
            value = self.get_value(node[i], edge[i])
            all_values[i] = value
        return all_values

    def get_batch_actions_probs(
            self,
            mini_batch_size,
            node: List[Dict[str, torch.Tensor]],
            edge: List[Dict[str, torch.Tensor]],
            centers_power_actions: list[torch.Tensor],
            center_func_actions: list[torch.Tensor],
    ):
        cf = [torch.zeros(0) for _ in range(mini_batch_size)]
        cp = [torch.zeros(0) for _ in range(mini_batch_size)]
        for i in range(mini_batch_size):
            _, _, _, log_power, log_funcs = self.get_action(node[i], edge[i], centers_power_actions[i],
                                                            center_func_actions[i])
            cf[i] = log_funcs
            cp[i] = log_power
        cf_l = torch.cat(cf, dim=0)
        cp_l = torch.cat(cp, dim=0)
        return cf_l, cp_l

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
            power_actions,
            func_actions,
            center_ratios,
            log_probs_power,
            log_probs_func
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
                mini_actions_power = [power_actions[i] for i in index]
                mini_actions_func = [func_actions[i] for i in index]
                mini_probs_func = [log_probs_func[i] for i in index]
                mini_probs_power = [log_probs_power[i] for i in index]
                new_pf, new_pp = self.get_batch_actions_probs(
                    mini_batch_size,
                    mini_nodes,
                    mini_edges,
                    mini_actions_power,
                    mini_actions_func,
                )
                # logger.debug(f"new_log_prob: {new_log_prob}")

                # 这里需要连接起来变成一个，要不然两个tensor不一样大
                ratios_0 = torch.mean(torch.exp(new_pf - torch.cat(mini_probs_func, dim=0)))
                ratios_1 = torch.mean(torch.exp(new_pp - torch.cat(mini_probs_power, dim=0)))
                ratios = (ratios_0 + ratios_1) / 2
                surr1 = ratios * flat_advantages[index]
                surr2 = torch.clamp(ratios, 1 - self.clip, 1 + self.clip)
                actor_loss: torch.Tensor = -torch.min(surr1, surr2)
                new_value = self.get_batch_values(
                    mini_nodes, mini_edges, mini_batch_size
                )
                critic_loss = F.mse_loss(flat_returns[index], new_value)
                total_loss: torch.Tensor = actor_loss.mean() + 0.5 * critic_loss
                self.optimizer.zero_grad()
                total_loss.backward()
                # 裁减
                # torch.nn.utils.clip_grad_norm_(self.network.parameters(), 10)
                self.optimizer.step()
                # batch_use_time = (datetime.now() - batch_time).microseconds

        return total_loss
