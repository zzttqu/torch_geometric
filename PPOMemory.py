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

        self.values = torch.zeros(batch_size).to(device)
        self.rewards = torch.zeros(batch_size).to(device)
        self.dones = torch.zeros(batch_size).to(device)
        self.time_step = torch.zeros(batch_size).to(device)
        self.count = 0

    def remember(
        self,
        node_state: Dict[str, torch.Tensor],
        edge_index: Dict[str, torch.Tensor],
        value: torch.Tensor,
        reward: float,
        done: int,
        total_action: torch.Tensor,
        log_probs: Dict[str, torch.Tensor],
        eposide_step: int = 0,
    ):
        self.node_states[self.count] = node_state
        self.total_actions[self.count] = total_action
        self.log_probs[self.count] = log_probs
        self.edge_indexs[self.count] = edge_index
        self.values[self.count] = value
        self.rewards[self.count] = reward
        self.dones[self.count] = done
        self.time_step[self.count] = eposide_step
        self.count += 1

    def generate_batches(
        self,
    ) -> (
        List[Dict[str, torch.Tensor]],
        List[Dict[str, torch.Tensor]],
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
