from torch.distributions import Categorical
from torch.utils.data import BatchSampler, SubsetRandomSampler

from GNNNet import GNNNet
import torch
import torch.nn.functional as F
from torch_geometric.data import Data


class PPOMemory:
    def __init__(self, batch_size, node_num, state_dim, action_dim):
        self.node_states = torch.zeros((batch_size, node_num, state_dim))
        self.edge_index = torch.zeros(batch_size)
        self.values = torch.zeros(batch_size)
        self.rewards = torch.zeros(batch_size)
        self.dones = torch.zeros(batch_size)
        self.total_actions = torch.zeros((batch_size, node_num * action_dim))
        self.log_probs = torch.zeros(batch_size)
        self.count = 0

    def remember(self, node_state, edge_index, value, reward, done, total_action, log_probs):
        self.node_states[self.count] = node_state
        self.edge_index[self.count] = edge_index
        self.values[self.count] = value
        self.rewards[self.count] = reward
        self.dones[self.count] = done
        self.total_actions[self.count] = total_action.view(-1)
        self.log_probs[self.count] = log_probs
        self.count += 1

    def generate_batches(self):
        self.count = 0
        return self.node_states, self.edge_index, self.values, self.rewards, self.dones, self.total_actions, self.log_probs


class Agent:
    def __init__(self, work_cell_num, center_num, gamma, batch_size, gae_lambda, n_epochs, mini_batch_size, clip=0.2,
                 lr=3e-4):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.work_cell_num = work_cell_num
        self.center_num = center_num
        self.gamma = gamma
        self.batch_size = batch_size
        self.gae_lambda = gae_lambda
        self.n_epochs = n_epochs
        self.network = GNNNet(node_num=6, state_dim=4, action_dim=2).double()
        self.mini_batch_size = mini_batch_size
        self.clip = clip
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=lr, eps=1e-5)

    def get_value(self, state, edge_index) -> torch.Tensor:
        data = Data(x=state, edge_index=edge_index)
        _, value = self.network(data)
        return value

    def get_action(self, state, edge_index, raw=None):
        data = Data(x=state, edge_index=edge_index)
        logits, _ = self.network(data)
        # 第一项是功能动作，第二项是是否接受上一级运输
        logits = logits.reshape((self.work_cell_num, 2))
        action_material_dist = Categorical(logits=logits)
        # 前半段是动作，后半段是接受动作
        raw = action_material_dist.sample()
        materials = raw[self.work_cell_num:]
        actions = raw[:self.work_cell_num]
        all_log_probs = torch.stack([categorical.log_prob(action) for action, categorical in
                                     zip(raw, action_material_dist)])
        return raw, all_log_probs.sum(0)

    def learn(self, ppo_memory: PPOMemory, last_node_state, last_done, total_steps):
        node_states, edge_index, values, rewards, dones, total_actions, log_probs = ppo_memory.generate_batches()
        flat_states = node_states.reshape(-1, node_states.shape[-1])
        flat_actions = total_actions.reshape(-1, total_actions.shape[-1])
        flat_probs = log_probs.reshape(-1, log_probs.shape[-1])
        with torch.no_grad():
            last_value = self.get_value(last_node_state, edge_index).reshape(1, -1)
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
                    delta = rewards[t] + self.gamma * next_values * next_nonterminal - values[t]
                    last_gae_lam = delta + self.gamma * self.gae_lambda * next_nonterminal * last_gae_lam
                    advantages[t] = last_gae_lam
            advantages = ((advantages - advantages.mean()) / (advantages.std() + 1e-5))
            # 这个是value的期望值
            returns = advantages + values
        flat_advantages = advantages.reshape(-1)
        flat_returns = returns.reshape(-1)
        for _ in range(self.n_epochs):
            for index in BatchSampler(SubsetRandomSampler(range(self.batch_size)), self.mini_batch_size, False):
                # 这里的index默认是一维的，没设置乱七八糟的
                _, new_log_prob = self.get_action(flat_states[index], edge_index[index], flat_actions[index])
                ratios = torch.exp(new_log_prob - flat_probs[index])
                surr1 = ratios * flat_advantages[index]
                surr2 = torch.clamp(ratios, 1 - self.clip, 1 + self.clip)
                actor_loss: torch.Tensor = -torch.min(surr1, surr2)
                new_value = self.get_value(flat_states[index], edge_index[index])
                critic_loss = F.mse_loss(flat_returns[index], new_value.view(-1))
                total_loss: torch.Tensor = actor_loss + 0.5 * critic_loss
                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()
