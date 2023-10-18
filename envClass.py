from enum import Enum
import numpy as np
import networkx as nx
from matplotlib import pyplot as plt
import torch
from typing import List, Dict, Tuple, Union

from WorkCenter import WorkCenter


class StateCode(Enum):
    workcell_ready = 0
    workcell_working = 1
    workcell_low_material = 2
    workcell_low_product = 3
    workcell_function_error = 4


from TransitCenter import TransitCenter
from WorkCell import WorkCell

plt.rcParams["font.sans-serif"] = ["SimHei"]  # 显示中文标签
plt.rcParams["axes.unicode_minus"] = False


def select_functions(start, end, work_center_num, fun_per_center):
    num_selections = work_center_num * fun_per_center
    # 创建一个包含范围内所有数字的列表
    numbers = np.arange(start, end + 1)

    # 计算还需要额外选取的次数
    remaining_selections = num_selections - len(numbers)

    # 如果还需要额外选取，就继续随机选取，确保每个数字都被选取至少一次
    if remaining_selections > 0:
        additional_selections = np.random.choice(numbers, size=remaining_selections)
        numbers = np.concatenate((numbers, additional_selections))
    np.random.shuffle(numbers)
    # 转变为2个功能一组

    return numbers.reshape(-1, fun_per_center)


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


class EnvRun:
    def __init__(
        self,
        work_center_num,
        fun_per_center,
        function_num,
        device,
        episode_step_max=256,
        product_goal=500,
    ):
        self.device = device
        self.edge_index: Dict[Tuple[str, str, str], torch.Tensor] = {}
        self.work_center_num = work_center_num
        self.work_cell_num = self.work_center_num * fun_per_center
        self.work_center_list: List[WorkCenter] = []
        self.function_num = function_num
        self.center_num = function_num
        self.center_state_num = 3
        # 随机生成一个2*n的矩阵，每列对应一个工作中心F
        self.function_matrix = select_functions(
            0,
            function_num - 1,
            work_center_num,
            fun_per_center,
        )
        self.work_cell_list: List[WorkCell] = []
        # 生成物流运输中心代号和中心存储物料的对应关系
        self.id_center: np.ndarray = np.zeros((self.center_num, 2), dtype=int)
        # 初始化工作中心
        for function_list in self.function_matrix:
            self.work_center_list.append(WorkCenter(function_list))
        # 各级别生产能力
        self.product_capacity = [0 for _ in range(self.function_num)]

        for i in range(self.function_num):
            for work_center in self.work_center_list:
                # 函数返回值的第二位是funcid，第一位是workcellid
                fl = np.array([sub[1] for sub in work_center.get_all_cell_func()])
                indices = np.where(fl == i)[0]
                if len(indices) > 0:
                    self.product_capacity[i] += work_center.get_cell_speed(indices)
        # 初始化集散中心
        self.center_list: List[TransitCenter] = []
        for i in range(function_num):
            self.center_list.append(TransitCenter(i))
            self.id_center[i] = i
        # 初始化生产中心
        # 产品数量
        self.step_products = np.zeros(function_num)
        self.total_products = np.zeros(function_num)
        self.function_group = self.get_function_group()
        # 奖励和完成与否
        self.reward = 0
        self.done = 0
        self.product_goal = product_goal
        # 一次循环前的step数量
        self.episode_step = 0
        self.episode_step_max = episode_step_max

    def show_graph(self):
        # 可视化
        graph = nx.DiGraph()
        for node in self.work_cell_list:
            graph.add_node(node._id, working=node.working)
        for center in self.center_list:
            graph.add_node(
                center.cell_id + self.work_cell_num, product=center.product_id
            )
        # 生成边
        for work_cell in self.work_cell_list:
            for center in self.center_list:
                cell_fun_id = work_cell.function
                product_id = center.product_id
                # 边信息
                # 从生产到中转
                if cell_fun_id == product_id:
                    # 可视化节点需要id不能重复的
                    graph.add_edge(work_cell._id, center.cell_id + self.work_cell_num)
                # 从中转到下一步
                if product_id == cell_fun_id - 1:
                    # 可视化节点需要id不能重复的
                    graph.add_edge(center.cell_id + self.work_cell_num, work_cell._id)

    def build_edge(self):
        center_index = ("work_cell", "work_cell_to_work_cell", "work_cell")
        product_index = ("work_cell", "work_cell_to_center", "center")
        material_index = ("center", "center_to_work_cell", "work_cell")
        self.edge_index[center_index] = torch.zeros((2, 0), dtype=torch.long)
        self.edge_index[product_index] = torch.zeros((2, 0), dtype=torch.long)
        self.edge_index[material_index] = torch.zeros((2, 0), dtype=torch.long)
        # 连接在workcenter中生成的边
        for work_center in self.work_center_list:
            center_edge, product_edge, material_edge = work_center.build_edge(
                id_center=self.id_center
            )
            # 需要按列拼接
            self.edge_index[center_index] = torch.cat(
                [self.edge_index[center_index], center_edge], dim=1
            )
            self.edge_index[product_index] = torch.cat(
                [self.edge_index[product_index], product_edge], dim=1
            )
            self.edge_index[material_index] = torch.cat(
                [self.edge_index[material_index], material_edge], dim=1
            )
        self.edge_index[center_index] = self.edge_index[center_index].to(self.device)
        self.edge_index[product_index] = self.edge_index[product_index].to(self.device)
        self.edge_index[material_index] = self.edge_index[material_index].to(
            self.device
        )

        return self.edge_index

        # 生成边
        # region
        for work_cell in self.work_cell_list:
            for center in self.center_list:
                cell_fun_id = work_cell.function
                product_id = center.product_id
                # 边信息
                # 从生产到中转
                if cell_fun_id == product_id:
                    self.edge_index["work_cell_to_center"].append(
                        [work_cell._id, center.cell_id]
                    )
                # 从中转到下一步
                if product_id == cell_fun_id - 1:
                    self.edge_index["center_to_work_cell"].append(
                        [center.cell_id, work_cell._id]
                    )
        for key, value in self.edge_index.items():
            value = np.array(value)
            self.edge_index[key] = torch.tensor(value, dtype=torch.int64).T.to(
                self.device
            )
        # self.edge_index = torch.tensor(np.array(graph.edges()), dtype=torch.int64).T
        # endregion

    def deliver_centers_material(self, workcell_get_material):
        #  计算有同一功能有几个节点要接收
        collect = np.zeros(self.work_cell_num)
        for indices in self.function_group:
            if workcell_get_material[indices].sum() != 0:
                softmax_value = 1 / (workcell_get_material[indices].sum())
            else:
                softmax_value = 0
            collect[indices] = softmax_value * workcell_get_material[indices]
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
        # 处理产品
        for work_cell in self.work_cell_list:
            # 取出所有物品，放入center中
            self.center_list[work_cell.function].putin_product(work_cell.products)
            # 当前步全部的product数量
            products[work_cell.function] += work_cell.products

            work_cell.transport(2, 0)
        self.step_products = products
        self.total_products += self.step_products
        # 根据是否接收物料的这个动作空间传递原料
        for work_cell in self.work_cell_list:
            # 如果为原料处理单元，function_id为0
            if work_cell.function == 0:
                work_cell.transport(3, work_cell.speed)
            else:
                self.center_list[work_cell.function - 1].moveout_product(
                    int(products[work_cell.function - 1] * collect[work_cell._id])
                )
                work_cell.transport(
                    3,
                    int(products[work_cell.function - 1] * collect[work_cell._id]),
                )
                # # 看看当前id在flat里边排第几个，然后把对应权重进行计算
                # collect = flatt[torch.where(work_cell.cell_id == flat_id)[0].item()]
                # # int会导致有盈余，但是至少不会发生没办法转移的情况
                # self.center_list[work_cell.function - 1].moveout_product(
                #     int(products[work_cell.function - 1] * collect))
                # work_cell.transport(3, int(products[work_cell.function - 1] * collect))

    # def update_work_cell(self, workcell_id, workcell_action, workcell_function=0):
    #     for _id, action, work_cell in zip(workcell_id, workcell_action, self.work_cell_list):
    #         work_cell.work(action)

    def update_all_work_cell(self, workcell_action, workcell_function=0):
        # 先检测workcell是否出现冲突，冲突则报错

        for action, work_cell in zip(workcell_action, self.work_cell_list):
            work_cell.work(action)

    def update_all(self, all_action: torch.Tensor):
        # 这里需要注意是raw顺序是一个节点，两个动作，不能这样拆分，需要重新折叠
        # 要先折叠，再切片
        all_action_fold = all_action.view((-1, 2))
        work_cell_action_slice = all_action_fold[: self.work_cell_num].numpy()
        self.update_all_work_cell(work_cell_action_slice[:, 0])
        self.deliver_centers_material(work_cell_action_slice[:, 1])

        # 额定扣血
        stable_reward = (
            -0.05
            * self.work_cell_num
            / self.function_num
            * max(self.episode_step / self.episode_step_max, 1 / 2)
        )

        # 生产有奖励，根据产品级别加分
        products_reward = 0
        for i, prod_num in enumerate(self.step_products):
            # 生产数量/该产品生产单元数量*生产产品类别/总产品类别
            products_reward += (
                0.005
                * max(prod_num, 0.1)
                / self.product_capacity[i]
                * (i + 1)
                / self.function_num
            )
        # 最终产物奖励，要保证这个产物奖励小于扣血
        goal_reward = max(self.total_products[-1], 0.1) / self.product_goal * 0.1
        self.reward += stable_reward
        self.reward += goal_reward
        self.reward += products_reward
        self.episode_step += 1
        self.done = 0
        # 超过步数
        if self.episode_step > self.episode_step_max:
            # self.reward -= 10
            self.done = 1
        # 完成任务目标
        if self.total_products[-1] > self.product_goal:
            self.reward += 5
            self.done = 1

    def get_obs(self):
        center_states = torch.zeros((self.center_num, self.center_state_num)).to(
            self.device
        )
        a = []
        for work_center in self.work_center_list:
            a += work_center.get_all_cell_state()
        # 按cellid排序，因为要构造数据结构
        sort_state = sorted(a, key=lambda x: x[0])
        work_cell_states = torch.stack(sort_state).to(self.device)

        for center in self.center_list:
            center_states[center.cell_id] = center.get_state()

        obs_states: Dict[str, torch.Tensor] = {
            "work_cell": work_cell_states,
            "center": center_states,
        }
        # 保证obsstates和edgeindex都转到cuda上

        return obs_states, self.edge_index, self.reward, self.done, self.episode_step

    def get_function_group(self):
        work_function = []
        for work_center in self.work_center_list:
            work_function += work_center.get_all_cell_func()
        # 排序
        sort_func = sorted(work_function, key=lambda x: x[0])
        work_function = torch.tensor(sort_func).T
        # 针对function的分类
        unique_labels = torch.unique(work_function)
        grouped_indices = [
            torch.where(work_function == label)[0] for label in unique_labels
        ]
        # 因为功能0是从原材料是无穷无尽的，所以不需要考虑不需要改变
        return grouped_indices

    def reset(self):
        self.total_products = np.zeros(self.function_num)
        self.step_products = np.zeros(self.function_num)
        self.reward = 0
        self.done = 0
        self.episode_step = 0
        for worker in self.work_cell_list:
            worker.reset_state()
        for center in self.center_list:
            center.product_num = 0
