import random
from enum import Enum
import numpy as np
import networkx as nx
from matplotlib import pyplot as plt
import torch
from typing import List
from torch_geometric.data import Data
from GNNNet import GNNNet

graph = nx.Graph()
plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文标签
plt.rcParams['axes.unicode_minus'] = False


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
    return normalized_array


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
        self.edge_ratio = 1
        self.health = 100
        self.working = None
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

    def work(self, action, action_detail):
        # 工作/继续工作
        if action == 1:
            if action_detail == self.working:
                pass
            else:
                self.set_work(action_detail)
            self.idle_time = 0
        # 停止工作
        elif action == 0:
            # self.working = None
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
        if self.function[2] <= self.function[1]:
            self.state = StateCode.workcell_low_material
        else:
            self.state = StateCode.workcell_working
        # 爆仓了
        if self.working is None:
            return
        if self.function[3] >= 1000:
            print('爆仓了')

    def reset_state(self):
        self.state = StateCode.workcell_ready
        self.working = None

    # 状态空间
    def get_state(self):
        if self.state == StateCode.workcell_working:
            return (self.state.value,
                    self.working,
                    self.function[1],
                    self.function[2],
                    )
        else:
            return (self.state.value,
                    self.function[0],
                    self.function[1],
                    self.function[2],
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
        return self.state.value, self.product_id, 10, self.product_num


class EnvRun:
    def __init__(self, step, work_cell_num, function_num, agv_cell_num=0):
        self.step = step
        self.work_cell_num = work_cell_num
        self.agv_cell_num = agv_cell_num
        self.work_cell_state_num = 4
        # 会生产几种类型
        self.function_num = function_num
        self.work_cell_list: List[WorkCell] = []
        for i in range(work_cell_num):
            self.work_cell_list.append(
                WorkCell(function_id=np.random.randint(0, function_num), speed=6,
                         position=[i, 0], materials=10)
            )
        self.center_list: List[TransitCenter] = []
        for i in range(function_num):
            self.center_list.append(
                TransitCenter(i)
            )
        self.products = np.zeros(function_num)
        self.edge_weight = edge_weight_init(self.get_work_cell_functions())
        for worker, ratio in zip(self.work_cell_list, self.edge_weight):
            worker.edge_ratio = ratio[0]

    def build_edge(self):
        # 理论上来说边建立好后就不会变化了
        graph = nx.DiGraph()
        # node_id = np.zeros((self.work_cell_num + self.function_num), dtype=int)
        index = 0
        for worker in self.work_cell_list:
            graph.add_node(worker.cell_id, state=worker.state.value, function=worker.function[0])
            index += 1
        for center in self.center_list:
            graph.add_node(center.cell_id, state=center.state.value, function=center.product_id)
            index += 1

        for work_cell in self.work_cell_list:
            for center in self.center_list:
                cell1_id = work_cell.function[0]
                product_id = center.product_id
                # 边信息
                # 从生产到中转
                if cell1_id == product_id:
                    graph.add_edge(work_cell.cell_id, center.cell_id,
                                   function=cell1_id, ratio=1)
                # 从中转到下一步
                if product_id == cell1_id - 1:
                    graph.add_edge(center.cell_id, work_cell.cell_id,
                                   function=product_id, ratio=work_cell.edge_ratio)
        return graph

    def update_material(self, workcell_material_ratio):
        products = np.zeros(self.function_num)
        for work_cell in self.work_cell_list:
            # 取出所有物品，然后清空
            self.center_list[work_cell.function[0]].putin_product(work_cell.function[3])
            products[work_cell.function[0]] += (work_cell.function[3] + self.products[work_cell.function[0]])
            work_cell.transport(2, 0)
            # 更新边权重
            work_cell.edge_ratio = workcell_material_ratio[work_cell.cell_id]
        # 根据边权重传递物料
        for ratio, work_cell in zip(
                workcell_material_ratio,
                self.work_cell_list):
            # if action == 2:
            if work_cell.function[0] == 0:
                work_cell.transport(3, 100)
            else:
                # int会导致有盈余，但是至少不会发生没办法转移的情况
                self.center_list[work_cell.function[0] - 1].moveout_product(
                    int(products[work_cell.function[0] - 1] * ratio))
                work_cell.transport(3, int(products[work_cell.function[0] - 1] * ratio))
        # elif action == 0:
        #     if work_cell.function[0] == 0:
        #         pass
        #     else:
        #         pass

    def update_work_cell(self, workcell_id, workcell_action, workcell_function):
        for _id, action, f_id, work_cell in zip(workcell_id, workcell_action, workcell_function, self.work_cell_list):
            work_cell.work(action, f_id)

    def get_obs(self):
        obs_states = np.zeros((self.work_cell_num + self.function_num, self.work_cell_state_num))
        for work_cell in self.work_cell_list:
            obs_states[work_cell.cell_id] = work_cell.get_state()
        for center in self.center_list:
            obs_states[center.cell_id] = center.get_state()
        # 额定扣血
        reward = -0.1
        # 生产一个有奖励
        reward += self.center_list[-1].product_num * 0.1

        return obs_states, reward

    def get_work_cell_functions(self):
        work_station_functions = np.zeros((self.work_cell_num, 1), dtype=int)
        for i, work_cell in enumerate(self.work_cell_list):
            work_station_functions[i] = work_cell.get_function()
        return work_station_functions

    def get_work_cell_actions(self):
        return np.array([0, 1])


if __name__ == '__main__':
    np.set_printoptions(precision=3, suppress=True)
    torch.set_printoptions(precision=3, sci_mode=False)
    function_num = 3
    env = EnvRun(1, 6, function_num)
    work_function = env.get_work_cell_functions()
    # print(work_function)
    weight = edge_weight_init(work_function).squeeze()
    work_action = env.get_work_cell_actions()
    random_function = [np.random.choice(row) for row in work_function]
    random_action = [np.random.choice(work_action) for i in range(function_num)]
    # 设置任务计算生产
    env.update_work_cell([0, 1, 2, 3, 4], [1, 1, 1, 1, 1], random_function)
    # 神经网络要输出每个工作站的工作，功能和传输比率
    # 迁移物料,2是正常接收，其他是不接收
    env.update_material(weight)
    obs_state, _ = env.get_obs()

    # print(obs_state, products)
    print("=========")
    env.update_work_cell([0, 1, 2, 3, 4], [1, 1, 1, 1, 1], random_function)
    env.update_material(weight)
    obs_state, _ = env.get_obs()
    # print(obs_state, products)
    print("=========")
    env.update_work_cell([0, 1, 2, 3, 4], [1, 1, 1, 1, 1], random_function)
    env.update_material(weight)
    obs_state, reward = env.get_obs()
    # print(obs_state, reward)
    graph = env.build_edge()
    pos = {}
    for i in range(3):
        pos[env.work_cell_list[i].cell_id] = env.work_cell_list[i].position
    node_states = nx.get_node_attributes(graph, 'state')
    node_function = nx.get_node_attributes(graph, 'function')
    edge_weight = nx.get_edge_attributes(graph, 'ratio')
    nodes = nx.nodes(graph)
    edges = nx.edges(graph)
    node_labels = {}
    edge_labels = {}
    for node in nodes:
        # 这里只用\n就可以换行了
        node_labels[node] = f'{node}节点：\n 状态：{node_states[node]} \n 功能：{node_function[node]}'
    for edge in edges:
        edge_labels[edge] = f'权重{edge_weight[edge]:.2f}'
    # print(node_labels)
    pos = nx.spring_layout(graph)
    nx.draw_networkx_nodes(graph, pos)
    nx.draw_networkx_labels(graph, pos, node_labels)
    # nx.draw_networkx_edges(graph, pos, connectionstyle="arc3,rad=0.2")
    nx.draw_networkx_edges(graph, pos)
    nx.draw_networkx_edge_labels(graph, pos, edge_labels)
    edge_weight = torch.tensor(list(nx.get_edge_attributes(graph, 'ratio').values()))
    edge_index = torch.tensor(np.array(graph.edges()), dtype=torch.int64).T
    data = Data(x=torch.tensor(obs_state, dtype=torch.float64), edge_index=edge_index, edge_attr=edge_weight)
    net = GNNNet(node_num=6, embed_dim=4).double()
    print(net(data))
    print(data.num_features)
    # plt.show()
