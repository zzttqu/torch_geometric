import torch
from torch_geometric.data import Data, Batch, HeteroData
from typing import Dict, List, Tuple


class PPOMemory:
    def __init__(
        self,
        batch_size: int,
        device,
    ):
        self.node_states = [{} for _ in range(batch_size)]
        self.total_actions = [{} for _ in range(batch_size)]
        self.edge_indexs = [{} for _ in range(batch_size)]
        self.log_probs = [{} for _ in range(batch_size)]
        """ for key, value in node_dic.items():
            self.node_states[key] = torch.zeros((batch_size, value[0], value[1])).to(
                device
            )
            self.total_actions[key] = torch.zeros(
                (batch_size, value[0] * action_dim)
            ).to(device)
            self.log_probs[key] = torch.zeros((batch_size, value[0] * action_dim)).to(
                device
            )
        for key, value in edge_dic.items():
            self.edge_indexs[key] = torch.zeros(
                (batch_size, 2, value), dtype=torch.int64
            ).to(device) """
        self.values = torch.zeros(batch_size).to(device)
        self.rewards = torch.zeros(batch_size).to(device)
        self.dones = torch.zeros(batch_size).to(device)
        self.count = 0

    def remember(
        self,
        node_state: Dict[str, torch.Tensor],
        edge_index: Dict[Tuple[str], torch.Tensor],
        value: torch.Tensor,
        reward: torch.Tensor,
        done: int,
        total_action: Dict[str, torch.Tensor],
        log_probs: Dict[str, torch.Tensor],
    ):
        self.node_states[self.count] = node_state
        self.total_actions[self.count] = total_action
        self.log_probs[self.count] = log_probs
        self.edge_indexs[self.count] = edge_index
        self.values[self.count] = value
        self.rewards[self.count] = reward
        self.dones[self.count] = done
        self.count += 1

    def generate_batches(
        self,
    ) -> (
        List[Dict[str, torch.Tensor]],
        List[Dict[Tuple[str, str, str], torch.Tensor]],
        torch.Tensor,
        torch.Tensor,
        List[Dict[str, torch.Tensor]],
        List[Dict[str, torch.Tensor]],
    ):
        data_list = []
        """ for i in range(self.count):
            hetero_data = HeteroData()
            # 节点信息
            for key, _value in self.node_states.items():
                hetero_data[key].x = _value[i]
            # 边信息
            for key, _value in self.edge_indexs.items():
                hetero_data[key].edge_index = _value[i]
            data_list.append(hetero_data)
            # data_list.append(Data(x=self.node_states[i], edge_index=self.edge_index[i]))
        batch = Batch.from_data_list(data_list) """

        # dataloader = DataLoader(data_list, batch_size=1, shuffle=False)
        self.count = 0
        return (
            self.node_states,
            self.edge_indexs,
            self.values,
            self.rewards,
            self.dones,
            self.total_actions,
            self.log_probs,
        )
