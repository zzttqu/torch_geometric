from enum import Enum
from itertools import count
from platform import node
import numpy as np
import networkx as nx
from matplotlib import pyplot as plt
import torch
from typing import List, Dict, Tuple, Union
from torch_geometric.data import HeteroData, Data
from torch_geometric.utils import to_networkx
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


def gen_pos(node_lists: List[List], nodes):
    """产生工作单元和搬运中心的位置坐标

    Args:
        node_list (_type_): _description_
        node_type_array (_type_): _description_

    Raises:
        SystemExit: _description_

    Returns:
        _type_: _description_
    """
    step = 1
    x_list = []
    y_list = []
    for i, node_list in enumerate(node_lists):
        x_step = 10 / len(node_list)
        for j in range(len(node_list)):
            y_list.append(i * step)
            x_list.append((j + 0.5) * x_step)

    pos = {}
    for node, x, y in zip(nodes, x_list, y_list):
        pos[node] = (x, y)
    return pos


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
        for work_center in self.work_center_list:
            fl = work_center.get_all_funcs()
            for i in range(self.function_num):
                # 函数返回值的是funcid
                # where返回的是tuple，需要提取第一项
                indices = np.where(fl == i)[0]
                # indices实际上是index，不是id
                if len(indices) > 0:
                    self.product_capacity[i] += work_center.get_cell_speed(indices)
        # 初始化集散中心
        self.storage_list: List[TransitCenter] = []
        for i in range(function_num):
            self.storage_list.append(TransitCenter(i))
            self.id_center[i] = i
        # 初始化生产中心
        # 产品数量
        self.step_products = np.zeros(function_num)
        self.function_group = self.get_function_group()
        # 奖励和完成与否
        self.reward = 0
        self.done = 0
        # 根据生产能力和最大步数计算生产目标数量
        # 根据水桶效应，选择最低的生产能力环节代表
        # TODO 每个工步的生产能力其实是波动的，因为其实是工作中心的生产能力
        desire_product_goal = int(0.2 * episode_step_max * min(self.product_capacity))
        if abs(product_goal - desire_product_goal) < 100:
            self.product_goal = desire_product_goal
        else:
            self.product_goal = desire_product_goal
        # 一次循环前的step数量
        self.episode_step = 0
        print("目标生产数量:", self.product_goal)

        self.episode_step_max = episode_step_max

    def show_graph(self, step):
        # norm_data: Data = data.to_homogeneous()
        process_label = {}
        process_edge = []
        # 这个size是有多少个worcell，主要是为了重新编号
        size = self.obs_states["work_cell"].shape[0]
        # 总节点数量
        count = 0
        for _key, _value in self.obs_states.items():
            # 按照顺序一个一个处理成字典和元组形式，state是一行的数据
            for i, state in enumerate(_value):
                # 根据键值不同设置不同的属性
                # cell的function指的是其生成的产品id
                state = list(map(int, state))
                if _key == "work_cell":
                    process_label[
                        count
                    ] = f"{i}：\n 状态：{state[1]} \n 功能：{state[0]} \n 属于{state[2]}"
                elif _key == "center":
                    process_label[
                        count
                    ] = f"{i}：\n 状态：{state[1]} \n 产品：{state[0]} \n 数量：{state[2]}"
                count += 1
        for _key, _edge in self.edge_index.items():
            node1, node2 = _key.split("_to_")
            edge_T = _edge.T.tolist()
            if node1 == "center":
                for __edge in edge_T:
                    process_edge.append((__edge[0] + size, __edge[1]))
            elif node2 == "center":
                for __edge in edge_T:
                    process_edge.append((__edge[0], __edge[1] + size))
            else:
                for __edge in edge_T:
                    process_edge.append((__edge[0], __edge[1]))
        # print(process_edge)
        # print("========")
        # print(process_label)
        # print(count)
        graph = nx.DiGraph()
        graph.add_nodes_from([i for i in range(count)])
        graph.add_edges_from(process_edge)
        nn = [[i for i in range(size)], [i for i in range(size, count)]]
        pos = gen_pos(nn, [i for i in range(count)])
        plt.figure(figsize=(15, 10))
        nx.draw_networkx_nodes(
            graph,
            pos=pos,
            nodelist=[i for i in range(size)],
            node_color="red",
        )
        nx.draw_networkx_nodes(
            graph,
            pos,
            nodelist=[i for i in range(size, count)],
            node_color="blue",
        )
        nx.draw_networkx_labels(graph, pos=pos, labels=process_label)
        nx.draw_networkx_edges(graph, pos, edgelist=process_edge, edge_color="black")
        # nx.draw(graph, with_labels=True)

        plt.savefig(f"./graph/{step}.png", dpi=800, bbox_inches="tight")
        plt.close()
        return
        process_edge = []
        node = []
        hetero_data = HeteroData()
        # 节点信息
        for key, _value in self.obs_states.items():
            hetero_data[key].x = _value
            # 边信息
        for key, _value in self.edge_index.items():
            node1, node2 = key.split("_to_")
            hetero_data[(f"{node1}", f"{key}", f"{node2}")].edge_index = _value
        graph: nx.Graph = to_networkx(data=hetero_data, to_undirected=False)
        for key in self.edge_index.keys():
            graph.add_edges_from([self.edge_index[key]])
        nx.draw(graph, with_labels=True)
        plt.show()
        # 可视化
        # graph = nx.DiGraph()
        # for node in self.work_cell_list:
        #     graph.add_node(node._id, working=node.get_function())
        # for center in self.center_list:
        #     graph.add_node(
        #         center.cell_id + self.work_cell_num, product=center.product_id
        #     )
        # # 生成边
        # for work_cell in self.work_cell_list:
        #     for center in self.center_list:
        #         cell_fun_id = work_cell.function
        #         product_id = center.product_id
        #         # 边信息
        #         # 从生产到中转
        #         if cell_fun_id == product_id:
        #             # 可视化节点需要id不能重复的
        #             graph.add_edge(work_cell._id, center.cell_id + self.work_cell_num)
        #         # 从中转到下一步
        #         if product_id == cell_fun_id - 1:
        #             # 可视化节点需要id不能重复的
        #             graph.add_edge(center.cell_id + self.work_cell_num, work_cell._id)

    def build_edge(self) -> Dict[str, torch.Tensor]:
        # center2_index = "center_to_cell"
        center1_index = "cell_to_center"
        product_index = "center_to_storage"
        material_index = "storage_to_cell"
        self.edge_index[center1_index] = torch.zeros((2, 0), dtype=torch.long)
        self.edge_index[product_index] = torch.zeros((2, 0), dtype=torch.long)
        self.edge_index[material_index] = torch.zeros((2, 0), dtype=torch.long)
        # 连接在workcenter中生成的边
        for work_center in self.work_center_list:
            center_edge, product_edge, material_edge = work_center.build_edge(
                id_center=self.id_center
            )
            # 需要按列拼接
            self.edge_index[center1_index] = torch.cat(
                [self.edge_index[center1_index], center_edge], dim=1
            )
            self.edge_index[product_index] = torch.cat(
                [self.edge_index[product_index], product_edge], dim=1
            )
            if material_edge is not None:
                self.edge_index[material_index] = torch.cat(
                    [self.edge_index[material_index], material_edge], dim=1
                )
        self.edge_index[center1_index] = self.edge_index[center1_index].to(self.device)
        self.edge_index[product_index] = self.edge_index[product_index].to(self.device)
        self.edge_index[material_index] = self.edge_index[material_index].to(
            self.device
        )

        return self.edge_index

        # 生成边
        # region
        for work_cell in self.work_cell_list:
            for center in self.storage_list:
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
        ratio = np.zeros(self.work_cell_num)
        for indices in self.function_group:
            if workcell_get_material[indices].sum() != 0:
                softmax_value = 1 / (workcell_get_material[indices].sum())
            else:
                softmax_value = 0
            ratio[indices] = softmax_value * workcell_get_material[indices]
        products = np.zeros(self.function_num)
        # 处理产品
        for work_center in self.work_center_list:
            # 这个是取出当前正在工作的
            _funcs: int = work_center.get_func()
            _products = work_center.get_product()
            # 取出所有物品，放入center中
            self.storage_list[_funcs].recive_product(work_center.get_product())
            # 当前步全部的product数量
            products[_funcs] += _products
            # 转移产品，清空workcell库存
            work_center.send_product()
        self.step_products = products
        # 根据是否接收物料的这个动作空间传递原料
        for work_center in self.work_center_list:
            id_funcs = np.array(work_center.get_all_cellid_func(), dtype=int)
            # 第一列是cellid，第二列是functionid
            # 我就是让他相乘一个系数，如果不分配，这个系数就是0
            for _func in id_funcs:
                # 第一位是cellid，第二位是functionid
                assert isinstance(_func, np.ndarray)
                _func_id: int = _func[1]
                _cell_id: int = _func[0]
                self.storage_list[_func_id].send_product(
                    int(products[_func_id] * ratio[_cell_id])
                )
                # 因为products和collect都是ndarray，可以使用列表直接获取元素
            # print(id_funcs, id_funcs[:, 0])
            #
            # collect是每个cell的权重
            work_center.recive_material(
                products[id_funcs[:, 1]] * ratio[id_funcs[:, 0]].tolist()
            )
            # # 看看当前id在flat里边排第几个，然后把对应权重进行计算
            # collect = flatt[torch.where(work_cell.cell_id == flat_id)[0].item()]
            # # int会导致有盈余，但是至少不会发生没办法转移的情况
            # self.center_list[work_cell.function - 1].moveout_product(
            #     int(products[work_cell.function - 1] * collect))
            # work_cell.transport(3, int(products[work_cell.function - 1] * collect))

    def update_all_work_center(self, workcell_action: np.ndarray):
        # 先检测workcell是否出现冲突，冲突则报错
        # 按照每个工作中心具有的工作单元的数量进行折叠

        for action, work_center in zip(workcell_action, self.work_center_list):
            work_center.work(int(action))

    def update_all(self, all_action: Dict[str, torch.Tensor]):
        # action按节点类别分开

        self.update_all_work_center(all_action["center"].numpy())
        # 针对workcenter的
        self.deliver_centers_material(all_action["cell"].numpy())

        # 额定扣血
        stable_reward = (
            -0.05
            * self.work_cell_num
            / self.function_num
            * max(self.episode_step / self.episode_step_max, 1 / 2)
        )

        # 生产有奖励，根据产品级别加分
        products_reward = 0
        # for i, prod_num in enumerate(self.step_products - 1):
        #     # 生产数量/该产品生产单元数量*生产产品类别/总产品类别，当生产第一个类别的时候不计数
        #     products_reward += (
        #         -0.002
        #         * prod_num
        #         / self.product_capacity[i]
        #         # * (i)
        #         # / self.function_num
        #     )
        # 最终产物肯定要大大滴加分
        products_reward += 0.05 * self.step_products[-1] / self.product_capacity[-1]
        # 最终产物奖励，要保证这个产物奖励小于扣血
        goal_reward = self.storage_list[-1].get_product_num() / self.product_goal * 0.01
        self.reward += stable_reward
        self.reward += goal_reward
        self.reward += products_reward
        self.done = 0
        self.episode_step += 1
        # 相等的时候刚好是episode，就是1,2,3,4，4如果是max，那等于4的时候就应该跳出了
        if self.episode_step >= self.episode_step_max:
            self.reward -= 5
            self.done = 1
        # 完成任务目标
        elif self.storage_list[-1].get_product_num() > self.product_goal:
            self.reward += 5
            self.done = 1

    def get_obs(
        self,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], float, int, int]:
        storage_states = torch.zeros(
            (self.center_num, self.center_state_num), dtype=torch.float
        ).to(self.device)
        a = []
        for work_center in self.work_center_list:
            a += work_center.get_all_cell_state()
        # 按cellid排序，因为要构造数据结构，其实不用排序，因为本来就是按顺序生成的。。。。
        # sort_state = sorted(a, key=lambda x: x[0])
        work_cell_states = torch.stack(a).to(self.device)

        for storage in self.storage_list:
            storage_states[storage.cell_id] = storage.get_state()
        b = []
        for center in self.work_center_list:
            b.append(center.get_state())
        work_center_states = torch.stack(b).to(self.device)

        self.obs_states: Dict[str, torch.Tensor] = {
            "cell": work_cell_states,
            "center": work_center_states,
            "storage": storage_states,
        }
        # 保证obsstates和edgeindex都转到cuda上

        return (
            self.obs_states,
            self.edge_index,
            self.reward,
            self.done,
            self.episode_step,
        )

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
        for center in self.storage_list:
            center.product_num = 0
