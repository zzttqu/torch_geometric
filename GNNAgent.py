from torch.distributions import Categorical

from GNNNet import GNNNet
import torch


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
    def __init__(self, work_cell_num, center_num, gamma):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.work_cell_num = work_cell_num
        self.center_num = center_num
        self.gamma = gamma
        self.network = GNNNet(node_num=6, state_dim=4, action_dim=2).double()

    def get_value(self, state, edge_index) -> torch.Tensor:
        _, value = self.network(state, edge_index)
        return value

    def get_action(self, state, edge_index, actions=None):
        logits, _ = self.network(state, edge_index)
        # 第一项是功能动作，第二项是是否接受上一级运输
        logits = logits.reshape((self.work_cell_num, 2))
        action_material_dist = Categorical(logits=logits)
        # 前半段是动作，后半段是接受动作
        raw = action_material_dist.sample()
        materials = raw[self.work_cell_num:]
        actions = raw[:self.work_cell_num]
        all_log_probs = torch.stack([categorical.log_prob(action) for action, categorical in
                                     zip(raw, action_material_dist)])
        return actions, materials, raw, all_log_probs.sum(0)

    def learn(self, ppo_memory: PPOMemory, last_node_state, last_done, total_steps):
        node_states, edge_index, values, rewards, dones, total_actions, log_probs = ppo_memory.generate_batches()
        flat_states = node_states.reshape(-1, node_states.shape[-1])
        flat_actions = total_actions.reshape(-1, total_actions.shape[-1])
        flat_probs = log_probs.reshape(-1, log_probs.shape[-1])
        with torch.no_grad():
            last_value = self.get_value(last_node_state, edge_index).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(self.device)
