import random
from enum import Enum
import numpy as np
import networkx as nx
from matplotlib import pyplot as plt
import torch
from typing import List

from torch.distributions import Categorical
from torch_geometric.data import Data
from GNNNet import GNNNet

graph = nx.Graph()
plt.rcParams["font.sans-serif"] = ["SimHei"]  # 显示中文标签
plt.rcParams["axes.unicode_minus"] = False

import random


def select_functions(start, end, num_selections):
    # 创建一个包含范围内所有数字的列表
    numbers = np.arange(start, end + 1)

    # 计算还需要额外选取的次数
    remaining_selections = num_selections - len(numbers)

    # 如果还需要额外选取，就继续随机选取，确保每个数字都被选取至少一次
    if remaining_selections > 0:
        additional_selections = np.random.choice(numbers, size=remaining_selections)
        np.concatenate((numbers, additional_selections))
    np.random.shuffle(numbers)

    return numbers


def edge_weight_init(raw_array):
    # 示例二维数组
    value_counts = {}
    for value in np.unique(raw_array):
        count = np.count_nonzero(raw_array == value)
        value_counts[value] = count
    total_count = len(raw_array)
    normalized_value = {}
    for value, count in value_counts.items():
        ratio = 1 / count if count != 0 else 0
        normalized_value[value] = ratio
    normalized_array = np.copy(raw_array)
    for value, ratio in normalized_value.items():
        normalized_array = np.where(normalized_array == value, ratio, normalized_array)
    return torch.tensor(normalized_array)


class StateCode(Enum):
    workcell_ready = 0
    workcell_working = 1
    workcell_low_material = 2
    workcell_low_product = 3
    # workcell_finish = 4
    AGVCell_ready = 0
    AGVCell_start = 1
    AGVCell_busy = 2
    AGVCell_low_battery = 3
    AGVCell_overload = 4


class WorkCell:
    next_id = 0

    def __init__(self, function_id, speed, position, materials=0, products=0):
        super().__init__()
        # 需要有当前这个工作单元每个功能的备件，每个功能生产效率
        self.cell_id = WorkCell.next_id
        WorkCell.next_id += 1
        self.function = np.array([function_id, speed, materials, products], dtype=int)
        self.position = np.array(position)
        self.health = 100
        self.working = False
        self.idle_time = 0
        self.state = StateCode.workcell_ready

    def set_work(self, function):
        self.working = function
        self.state = StateCode.workcell_working

    def transport(self, action, num):
        # 转移生产产品
        if action == 2:
            self.function[3] = 0

        # 或者接收原材料
        elif action == 3:
            self.function[2] += num

    def work(self, action):
        # 工作/继续工作
        if action == 1:
            if self.working:
                pass
            else:
                self.set_work(self.function[0])
                self.working = self.function[0]
            self.idle_time = 0
        # 停止工作
        elif action == 0:
            # self.working = None
            self.working = None
            self.state = StateCode.workcell_ready
        # 检查当前状态
        self.state_check()
        if self.state == StateCode.workcell_working:
            # 工作中
            # 2是生产库存数量，0是生产速度,1是原料数量
            self.function[3] += self.function[1]
            self.function[2] -= self.function[1]
            # self.health -= 0.1
        if self.state == StateCode.workcell_ready:
            self.idle_time += 1

    def state_check(self):
        # 低健康度
        # if self.health < 50:
        #     self.working = None
        #     self.state = StateCode.workcell_low_health
        # 缺少原料
        if self.function[2] < self.function[1]:
            self.state = StateCode.workcell_low_material
            self.working = None
        # 不缺货就变为ready状态
        elif (
            self.function[2] >= self.function[1]
            and self.state == StateCode.workcell_low_material
        ):
            self.state = StateCode.workcell_ready
        # 爆仓了
        if self.working is None:
            return

    def reset_state(self):
        self.state = StateCode.workcell_ready
        self.function[2] = self.function[1]
        self.function[3] = 0
        self.working = None

    # 状态空间
    def get_state(self):
        if self.state == StateCode.workcell_working:
            return torch.tensor(
                [self.state.value, self.working, self.function[1], self.function[2]]
            )

        else:
            return torch.tensor(
                [self.state.value, self.function[0], self.function[1], self.function[2]]
            )

    # 动作空间
    def get_function(self):
        return self.function[0]


class AGVCell:
    def __init__(self, speed, capacity):
        self.speed = speed
        self.capacity = capacity
        self.health = 100
        self.battery = 100
        self.working = False
        # 运行到指定位置所需时间
        self.running_time = 0
        self.location = np.array([1, 2])

    def set_work(self, goal, payload):
        # 应该允许临时超载
        if payload > self.capacity:
            return StateCode.AGVCell_overload
        distance = np.sum(np.abs(goal - self.location))
        self.running_time = distance / self.speed
        # 如果电量不支持抵达就报错
        if self.running_time > self.battery:
            return StateCode.AGVCell_low_battery
        self.working = True
        return StateCode.AGVCell_start

    def work(self):
        if self.running_time > self.battery:
            return StateCode.AGVCell_low_battery
        self.running_time -= 1
        self.battery -= 1
        return StateCode.AGVCell_busy

    def state_check(self):
        if self.running_time > self.battery:
            return StateCode.AGVCell_low_battery
        if self.working:
            return StateCode.AGVCell_busy


class TransitCenter:
    def __init__(self, product_id):
        self.cell_id = WorkCell.next_id
        WorkCell.next_id += 1
        self.product_id = product_id
        self.state = StateCode.workcell_working
        self.product_num = 0

    def putin_product(self, num):
        self.product_num += num
        return self.product_num

    def moveout_product(self, num):
        self.product_num -= num
        return self.product_num

    def get_state(self):
        return torch.tensor((self.state.value, self.product_id, 10, self.product_num))


class EnvRun:
    def __init__(self, work_cell_num, function_num, device):
        self.device = device
        self.edge_index = None
        self.work_cell_num = work_cell_num
        self.work_cell_state_num = 4
        # 会生产几种类型
        self.function_num = function_num
        self.work_cell_list: List[WorkCell] = []
        for i in range(work_cell_num):
            self.work_cell_list.append(
                # 随机function或者规定
                # i // (self.work_cell_num // self.function_num)
                # np.random.randint(0, function_num)
                # TODO 需要修改为random
                WorkCell(
                    function_id=select_functions(0, function_num, self.work_cell_num),
                    speed=6,
                    position=[i, 0],
                    materials=10,
                )
            )
        # 集散中心
        self.center_list: List[TransitCenter] = []
        for i in range(function_num):
            self.center_list.append(TransitCenter(i))
        self.products = np.zeros(function_num)
        self.function_group = self.get_function_group()

    def build_edge(self):
        # 理论上来说边建立好后就不会变化了
        graph = nx.DiGraph()
        # node_id = np.zeros((self.work_cell_num + self.function_num), dtype=int)
        index = 0
        for worker in self.work_cell_list:
            graph.add_node(
                worker.cell_id, state=worker.state.value, function=worker.function[0]
            )
            index += 1
        for center in self.center_list:
            graph.add_node(
                center.cell_id, state=center.state.value, function=center.product_id
            )
            index += 1

        for work_cell in self.work_cell_list:
            for center in self.center_list:
                cell1_id = work_cell.function[0]
                product_id = center.product_id
                # 边信息
                # 从生产到中转
                if cell1_id == product_id:
                    graph.add_edge(work_cell.cell_id, center.cell_id)
                # 从中转到下一步
                if product_id == cell1_id - 1:
                    graph.add_edge(center.cell_id, work_cell.cell_id)
        self.edge_index = torch.tensor(np.array(graph.edges()), dtype=torch.int64).T
        return graph

    def update_centers(self, workcell_get_material):
        # workcell_get_material = torch.tensor([1, 1, 1, 1, 1, 1], dtype=torch.float)
        #  处理边数据
        collect = torch.zeros(self.work_cell_num)
        for indices in self.function_group:
            if workcell_get_material[indices].sum() != 0:
                softmax_value = 1 / (workcell_get_material[indices].sum())
            else:
                softmax_value = 0
            collect[indices] = softmax_value * workcell_get_material[indices].float()
            # print(f"总共有{workcell_get_material[indices]}")
            # print(f"softmax_value{softmax_value}")
            # print(indices)
        # flatt = torch.cat(softmax_values).squeeze()
        # flat_id = torch.cat(self.function_group)
        # # print(flatt, flat_id)
        # # edgeindex第二行是接收方，选择出除了function0的节点id在接收方的
        # any1 = torch.isin(self.edge_index[1], flat_id)
        # edge_weight_index = torch.where(any1)[0]
        # # 重新给edgeweigh赋值
        # for i, index in enumerate(edge_weight_index):
        #     self.edge_weight[index] = flatt[i]
        products = np.zeros(self.function_num)
        for work_cell in self.work_cell_list:
            # 取出所有物品，然后清空
            self.center_list[work_cell.function[0]].putin_product(work_cell.function[3])
            products[work_cell.function[0]] += (
                work_cell.function[3] + self.products[work_cell.function[0]]
            )
            work_cell.transport(2, 0)
        # 根据边权重传递物料
        for work_cell in self.work_cell_list:
            # 如果为原料处理单元，function_id为0
            if work_cell.function[0] == 0:
                work_cell.transport(3, work_cell.function[1])
            else:
                self.center_list[work_cell.function[0] - 1].moveout_product(
                    int(
                        products[work_cell.function[0] - 1] * collect[work_cell.cell_id]
                    )
                )
                work_cell.transport(
                    3,
                    int(
                        products[work_cell.function[0] - 1] * collect[work_cell.cell_id]
                    ),
                )
                # # 看看当前id在flat里边排第几个，然后把对应权重进行计算
                # collect = flatt[torch.where(work_cell.cell_id == flat_id)[0].item()]
                # # int会导致有盈余，但是至少不会发生没办法转移的情况
                # self.center_list[work_cell.function[0] - 1].moveout_product(
                #     int(products[work_cell.function[0] - 1] * collect))
                # work_cell.transport(3, int(products[work_cell.function[0] - 1] * collect))

    # def update_work_cell(self, workcell_id, workcell_action, workcell_function=0):
    #     for _id, action, work_cell in zip(workcell_id, workcell_action, self.work_cell_list):
    #         work_cell.work(action)

    def update_all_work_cell(self, workcell_action, workcell_function=0):
        for action, work_cell in zip(workcell_action, self.work_cell_list):
            work_cell.work(action)

    def update_all(self, raw: torch.Tensor):
        centers = raw[self.work_cell_num :]
        work_cells = raw[: self.work_cell_num]
        self.update_all_work_cell(work_cells)
        self.update_centers(centers)

    def get_obs(self):
        obs_states = torch.zeros(
            (self.work_cell_num + self.function_num, self.work_cell_state_num),
            dtype=torch.float64,
        )
        for work_cell in self.work_cell_list:
            obs_states[work_cell.cell_id] = work_cell.get_state()
        for center in self.center_list:
            obs_states[center.cell_id] = center.get_state()
        # 额定扣血
        reward = -0.1
        # 生产一个有奖励
        # reward += self.center_list[-1].product_num * 0.1
        # 构造边和节点
        done = 0
        if self.center_list[-1].product_num > 50:
            reward += 10
            done = 1
        device_state = obs_states.to(self.device)
        device_edge = self.edge_index.to(self.device)
        device_reward = reward

        return device_state, device_edge, device_reward, done

    def get_work_cell_functions(self):
        work_station_functions = np.zeros((self.work_cell_num, 1), dtype=int)
        for i, work_cell in enumerate(self.work_cell_list):
            work_station_functions[i] = work_cell.get_function()
        return work_station_functions

    def get_function_group(self):
        # 针对function的分类
        work_function = self.get_work_cell_functions()
        function_id = torch.tensor(work_function.squeeze())
        unique_labels = torch.unique(function_id)
        grouped_indices = [
            torch.where(function_id == label)[0] for label in unique_labels
        ]
        # 因为功能0是从原材料是无穷无尽的，所以不需要考虑不需要改变
        return grouped_indices

    def reset(self):
        for worker in self.work_cell_list:
            worker.reset_state()
        for center in self.center_list:
            center.product_num = 0


if __name__ == "__main__":
    np.set_printoptions(precision=3, suppress=True)
    torch.set_printoptions(precision=3, sci_mode=False)
    function_num = 3
    work_cell_num = 6
    env = EnvRun(
        1,
        work_cell_num=work_cell_num,
        function_num=function_num,
        device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    )
    graph = env.build_edge()
    work_function = env.get_work_cell_functions()
    weight = torch.tensor([1] * work_cell_num, dtype=torch.float)
    random_function = [row[0] for row in work_function]
    # 设置任务计算生产
    env.update_all(torch.tensor([0, 1, 2, 3, 4, 5, 1, 1, 1, 1, 1, 1]))
    # 神经网络要输出每个工作站的工作，功能和传输比率
    # 迁移物料,2是正常接收，其他是不接收
    env.update_centers(weight)

    # print(obs_state)

    # 可视化
    node_states = nx.get_node_attributes(graph, "state")
    node_function = nx.get_node_attributes(graph, "function")
    nodes = nx.nodes(graph)
    edges = nx.edges(graph)
    node_labels = {}
    edge_labels = {}
    for node in nodes:
        # 这里只用\n就可以换行了
        node_labels[
            node
        ] = f"{node}节点：\n 状态：{node_states[node]} \n 功能：{node_function[node]}"

    # print(node_labels)
    pos = nx.spring_layout(graph)
    nx.draw_networkx_nodes(graph, pos)
    nx.draw_networkx_labels(graph, pos, node_labels)
    # nx.draw_networkx_edges(graph, pos, connectionstyle="arc3,rad=0.2")
    nx.draw_networkx_edges(graph, pos)

    device_state, device_edge, device_reward, done = env.get_obs()

    # print(obs_state)

    plt.show()
