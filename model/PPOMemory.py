from typing import Dict, List, Tuple

import torch
from torch import Tensor


class PPOMemory:
    def __init__(
            self,
            batch_size: int,
            device,
    ):
        self.node_states = [{} for _ in range(batch_size)]
        self.edge_indexes = [{} for _ in range(batch_size)]
        # 这里的actions用字典来存不同类型的动作
        self.power_actions = [torch.zeros(0) for _ in range(batch_size)]
        self.func_actions = [torch.zeros(0) for _ in range(batch_size)]
        self.center_ratios = [torch.zeros(0) for _ in range(batch_size)]
        self.log_probs_power = [torch.zeros(0) for _ in range(batch_size)]
        self.log_probs_func = [torch.zeros(0) for _ in range(batch_size)]
        self.values = torch.zeros(batch_size, device=device)
        self.rewards = torch.zeros(batch_size, device=device)
        self.dones = torch.zeros(batch_size, device=device)
        self.time_step = torch.zeros(batch_size, device=device)
        self.count = 0

    def remember(
            self,
            node_state: Dict[str, Tensor],
            edge_index: Dict[str, Tensor],
            value: Tensor,
            reward: float,
            done: int,
            power_actions: Tensor,
            func_actions: Tensor,
            center_ratios: Tensor,
            log_probs_power: Tensor,
            log_probs_func: Tensor,
            episode_step: int = 0,
    ):
        self.node_states[self.count] = node_state
        self.power_actions[self.count] = power_actions
        self.func_actions[self.count] = func_actions
        self.center_ratios[self.count] = center_ratios
        self.log_probs_power[self.count] = log_probs_power
        self.log_probs_func[self.count] = log_probs_func
        self.edge_indexes[self.count] = edge_index
        self.values[self.count] = value
        self.rewards[self.count] = reward
        self.dones[self.count] = done
        self.time_step[self.count] = episode_step
        self.count += 1

    def generate_batches(
            self,
    ) -> Tuple[
        List[Dict[str, Tensor]],
        List[Dict[str, Tensor]],
        Tensor,
        Tensor,
        Tensor,
        List[Tensor],
        List[Tensor],
        List[Tensor],
        List[Tensor],
        List[Tensor],
    ]:
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
            self.edge_indexes,
            self.values,
            self.rewards,
            self.dones,
            self.power_actions,
            self.func_actions,
            self.center_ratios,
            self.log_probs_power,
            self.log_probs_func,
        )
