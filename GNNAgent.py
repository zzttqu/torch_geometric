import os
from torch.distributions import Categorical
from torch.utils.data import BatchSampler, SubsetRandomSampler
from torch_geometric.data import Data, Batch, HeteroData
from torch_geometric.loader import DataLoader
from torch_geometric.nn import to_hetero, MetaPath2Vec
from GNNNet import GNNNet, HGTNet
import torch
import torch.nn.functional as F
import torch_geometric.transforms as T

#T.Constant()


class PPOMemory:
    def __init__(
        self,
        batch_size,
        node_dic: dict,
        edge_dic,
        state_dim,
        action_dim,
        device,
    ):
        self.node_states = {}
        self.total_actions = {}
        self.edge_indexs = {}
        self.log_probs = {}
        for key, value in node_dic.items():
            self.node_states[key] = torch.zeros((batch_size, value, state_dim)).to(
                device
            )
            self.total_actions[key] = torch.zeros((batch_size, value * action_dim)).to(
                device
            )
            self.log_probs[key] = torch.zeros((batch_size, value * action_dim)).to(
                device
            )
        for key, value in edge_dic.items():
            self.edge_indexs[key] = torch.zeros((batch_size, 2, value)).to(device)
        self.values = torch.zeros(batch_size).to(device)
        self.rewards = torch.zeros(batch_size).to(device)
        self.dones = torch.zeros(batch_size).to(device)
        self.count = 0

    def remember(
        self,
        node_state: dict,
        edge_index: dict,
        value: torch.Tensor,
        reward: torch.Tensor,
        done: int,
        total_action: dict,
        log_probs: dict,
    ):
        for key, value in node_state.items():
            self.node_states[key][self.count] = value
        for key, value in total_action.items():
            self.total_actions[key][self.count] = value
        for key, value in log_probs.items():
            self.log_probs[key][self.count] = value
        for key, value in edge_index.items():
            self.edge_indexs[key][self.count] = value
        self.values[self.count] = value
        self.rewards[self.count] = reward
        self.dones[self.count] = done

        self.count += 1

    def generate_batches(self):
        data_list = []
        for i in range(self.count):
            hetero_data = HeteroData()
            # 节点信息
            for key in self.node_states.keys:
                hetero_data[key].x = self.node_states[key][i]
            # 边信息
            for key in self.edge_indexs.keys:
                node1, node2 = key.split("_to_")
                hetero_data[node1, key, node2].edge_index = self.edge_indexs[key][i]
            data_list.append(hetero_data)
            # data_list.append(Data(x=self.node_states[i], edge_index=self.edge_index[i]))
        batch = Batch.from_data_list(data_list)

        dataloader = DataLoader(data_list, batch_size=1, shuffle=False)
        self.count = 0
        return (
            batch,
            self.values,
            self.rewards,
            self.dones,
            self.total_actions,
            self.log_probs,
        )


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
        assert init_data is not None, "init_data is None"

        # TODO 需要添加加载训练过的模型的代码和训练embedding代码
        self.update_heterodata(init_data)
        self.network = HGTNet(
            2,
            self.undirect_data,
        ).to(self.device)
        """self.embedding = MetaPath2Vec(
            edge_index_dict=self.undirect_data.edge_index_dict,
            embedding_dim=10,
            walk_length=1,
            context_size=1,
            metapath=self.metadata[1],
        )"""
        # 使用to_hetro相当于变了一个模型，还得todevice
        """self.network = to_hetero(self.network, self.metadata, aggr="sum").to(
            self.device
        )"""
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=lr, eps=1e-5)

    def update_heterodata(self, data: HeteroData):
        """更新heterodata"""
        self.undirect_data = T.ToUndirected()(data)
        # metadata函数第一个返回值是节点列表，第二个返回值是边列表
        self.metadata = self.undirect_data.metadata()

    """def get_embedding(self, data: HeteroData):
        for node_type in self.metadata[0]:
            print(data[node_type])
            print(self.embedding(node_type).size())
        raise SystemExit"""

    def get_value(self, state: dict, edge_index: dict) -> torch.Tensor:
        """如果不是学习状态就只能构造一下data"""
        hetero_data = HeteroData()
        # 节点信息
        for key, value in state.items():
            hetero_data[key].x = value
        # 边信息
        for key, value in edge_index.items():
            node1, node2 = key.split("_to_")
            hetero_data[node1, key, node2].edge_index = value
        # data = Data(x=state, edge_index=edge_index)
        _, value = self.network(T.ToUndirected()(hetero_data))
        return value

    def get_batch_values(self, batches):
        all_values = []
        for data in batches:
            _, value = self.network(data)
            all_values.append(value)
        all_values = torch.stack(all_values)
        return all_values

    def get_action(self, state: dict, edge_index: dict, raw=None):
        hetero_data = HeteroData()
        # 节点信息
        for key, value in state.items():
            hetero_data[key].x = value
        # 边信息
        for key, value in edge_index.items():
            node1, node2 = key.split("_to_")
            hetero_data[node1, key, node2].edge_index = value

        hetero_data = T.ToUndirected()(hetero_data)

        """ 如果不是学习状态就只能构造一下data """

        logits, _ = self.network(hetero_data.x_dict, hetero_data.edge_index_dict)
        print(logits)
        raise SystemExit
        # 前work_cell_num是
        # print(logits[0 : self.work_cell_num].shape)
        # 第一项是功能动作，第二项是是否接受上一级运输
        # 目前不需要对center进行动作，所以暂时不要后边的内容
        logits = logits[0 : self.work_cell_num]
        logits = logits.view((-1, 2))
        action_material_dist = Categorical(logits=logits)
        # 前半段是动作，后半段是接受动作
        if raw is None:
            raw = action_material_dist.sample()
        # materials = raw[self.work_cell_num :]
        # actions = raw[: self.work_cell_num]
        all_log_probs = torch.stack([action_material_dist.log_prob(raw)])

        return raw, all_log_probs.sum(0)

    def get_batch_actions_probs(self, batches: Batch, raw):
        all_logits = []
        for data in batches:
            logits, _ = self.network(data)
            logits = logits[:][0 : self.work_cell_num]
            # print(logits.shape,111)
            # 第一项是功能动作，第二项是是否接受上一级运输
            logits = logits.view((-1, 2))
            all_logits.append(logits)
        all_logits = torch.cat(all_logits, dim=0)
        action_material_dist = Categorical(logits=logits)
        all_log_probs = torch.stack([action_material_dist.log_prob(raw)])

        return all_log_probs.sum(0)

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
    ):
        (
            batches,
            values,
            rewards,
            dones,
            total_actions,
            log_probs,
        ) = ppo_memory.generate_batches()
        # flat_states = batches.reshape(-1, batches.shape[-1])
        flat_actions = total_actions.view(-1, total_actions.shape[-1])
        flat_probs = log_probs.view(-1, log_probs.shape[-1])
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
            returns = advantages + values
        flat_advantages = advantages.view(-1)
        flat_returns = returns.view(-1)
        total_loss = torch.tensor(0.0)
        for _ in range(self.n_epochs):
            # 这里是设置minibatch，也就是送入图神经网络的大小
            for index in BatchSampler(
                SubsetRandomSampler(range(self.batch_size)), 16, False
            ):
                new_log_prob = self.get_batch_actions_probs(
                    Batch.index_select(batches, index), total_actions[index]
                )
                ratios = torch.exp(new_log_prob - flat_probs[index]).sum(1)
                surr1 = ratios * flat_advantages[index]
                surr2 = torch.clamp(ratios, 1 - self.clip, 1 + self.clip)
                actor_loss: torch.Tensor = -torch.min(surr1, surr2)
                new_value = self.get_batch_values(batches[index])
                critic_loss = F.mse_loss(flat_returns[index], new_value.view(-1))
                total_loss: torch.Tensor = actor_loss.mean() + 0.5 * critic_loss
                self.optimizer.zero_grad()
                total_loss.backward()
                # 裁减
                # torch.nn.utils.clip_grad_norm_(self.network.parameters(), 10)
                self.optimizer.step()
        return total_loss
