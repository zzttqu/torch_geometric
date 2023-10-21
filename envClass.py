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
        self.edge_index: Dict[str, torch.Tensor] = {}
        self.work_center_num = work_center_num
        self.work_cell_num = self.work_center_num * fun_per_center
        self.func_per_center = fun_per_center
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
        # 这个初始化的顺序和工作单元的id顺序也是一致
        for function_list in self.function_matrix:
            self.work_center_list.append(WorkCenter(function_list))
        # 各级别生产能力
        self.product_capacity = [0 for _ in range(self.function_num)]

        for i in range(self.function_num):
            for work_center in self.work_center_list:
                # 函数返回值的第二位是funcid，第一位是workcellid
                fl = work_center.get_all_funcs()
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
            graph.add_node(node._id, working=node.get_function())
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
        center_index = "work_cell_to_work_cell"
        product_index = "work_cell_to_center"
        material_index = "center_to_work_cell"
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

    def deliver_centers_material(self, workcell_get_material: np.ndarray):
        # 计算有同一功能有几个节点要接收
        # 这个是每个id的cell的分配数量
        collect = np.zeros(self.work_cell_num)
        print(workcell_get_material)
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
        for work_center in self.work_center_list:
            # 这个是取出当前正在工作的
            _funcs = work_center.get_func()
            _products = work_center.get_product()
            # 取出所有物品，放入center中
            self.center_list[_funcs].recive_product(work_center.get_product())
            # 当前步全部的product数量
            products[_funcs] += _products
            # 转移产品，清空workcell库存
            work_center.send_product()
        self.step_products = products
        self.total_products += self.step_products
        # 根据是否接收物料的这个动作空间传递原料
        for work_center in self.work_center_list:
            id_funcs = np.array(work_center.get_all_cellid_func(), dtype=int)
            # 第一列是cellid，第二列是functionid
            # print(id_funcs[:, 0])
            # 如果为原料处理单元，function_id为0
            # 还需要考虑如果一个cell是0另一个不为0咋办。。。。
            if 0 in id_funcs[0, :]:
                zero_list = [
                    i for i, x in zip(id_funcs[:, 0], id_funcs[:, 1]) if x == 0
                ]
                # 如何function_id是0在workcenter中的话，就存入列表中，然后去workcenter中处理
                for zero_id in zero_list:
                    work_center.recive_material(zero_id)
            # 我就是让他相乘一个系数，如果不分配，这个系数就是0
            else:
                # 这个是center中的每一个单元的功能
                # 第一位是cellid，第二位是functionid
                for _func in id_funcs:
                    assert isinstance(_func, np.ndarray)
                    _func_id: int = _func[1]
                    _cell_id: int = _func[0]
                    self.center_list[_func_id].send_product(
                        int(products[_func_id] * collect[_cell_id])
                    )
                # 因为products和collect都是ndarray，可以使用列表直接获取元素
                work_center.recive_material(
                    [products[id_funcs[0, :]] * collect[work_center.get_all_cell_id()]]
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

    def update_all_work_center(self, workcell_action: np.ndarray, workcell_function=0):
        # 先检测workcell是否出现冲突，冲突则报错
        # 按照每个工作中心具有的工作单元的数量进行折叠
        workcell_action = workcell_action.reshape((-1, self.func_per_center))

        for action, work_center in zip(workcell_action, self.work_center_list):
            work_center.work(action)

    def update_all(self, all_action: torch.Tensor):
        # 这里需要注意是raw顺序是一个节点，两个动作，不能这样拆分，需要重新折叠
        # 要先折叠，再切片
        all_action_fold = all_action.view((-1, 2))
        work_cell_action_slice = all_action_fold[: self.work_cell_num].numpy()

        self.update_all_work_center(work_cell_action_slice[:, 0])
        # TODO 需要修改针对workcenter的
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

    def get_obs(
        self,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], float, int, int]:
        center_states = torch.zeros(
            (self.center_num, self.center_state_num), dtype=torch.float
        ).to(self.device)
        a = []
        for work_center in self.work_center_list:
            a += work_center.get_all_cell_state()
        # 按cellid排序，因为要构造数据结构
        sort_state = sorted(a, key=lambda x: x[0])
        work_cell_states = torch.stack(sort_state).float().to(self.device)

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
            work_function += work_center.get_all_cellid_func()
        # 排序
        sort_func = sorted(work_function, key=lambda x: x[0])
        work_function = np.array(sort_func).T
        # 针对function的分类
        unique_labels = np.unique(work_function[1])
        # 如果直接用where的话，返回的就是符合要求的index，
        # 刚好是按照cellid排序的，所以index就是cellid
        grouped_indices = [
            np.where(work_function[1] == label) for label in unique_labels
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
