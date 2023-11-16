from typing import List, Dict, Tuple

import networkx as nx
import numpy as np
import torch
from model.WorkCenter import WorkCenter
from model.StorageCenter import StorageCenter
from loguru import logger
from matplotlib import pyplot as plt
# from torch_geometric.data import HeteroData


# from model.WorkCell import WorkCell

# plt.rcParams["font.sans-serif"] = ["SimHei"]  # 显示中文标签
# plt.rcParams["axes.unicode_minus"] = False
import warnings


def deprecated(func):
    def wrapper(*args, **kwargs):
        warnings.warn("Call to deprecated function.", category=DeprecationWarning)
        return func(*args, **kwargs)

    return wrapper


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


@deprecated
def edge_weight_init(raw_array):
    # 示例二维数组
    value_counts = {}
    for value in np.unique(raw_array):
        count = np.count_nonzero(raw_array == value)
        value_counts[value] = count
    normalized_value = {}
    for value, count in value_counts.items():
        ratio = 1 / count if count != 0 else 0
        normalized_value[value] = ratio
    normalized_array = np.copy(raw_array)
    for value, ratio in normalized_value.items():
        normalized_array = np.where(normalized_array == value, ratio, normalized_array)
    return torch.tensor(normalized_array)


@deprecated
def gen_pos(node_lists: List[List], nodes):
    """产生工作单元和搬运中心的位置坐标

    Args:
        node_lists:
        nodes: _description_

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


def process_raw_data(raw_edge_index, raw_state) -> dict:
    process_edge = []
    process_states = []
    size = []
    count = 0
    # 处理state
    for _key, _value in raw_state.items():
        size.append(len(_value))
        # 按照顺序一个一个处理成字典和元组形式，state是一行的数据
        for i, state in enumerate(_value):
            # 根据键值不同设置不同的属性
            # cell的function指的是其生成的产品id
            # json字符串的话，process_state是一个数组一个是一个字典，包括id，name，category，
            # value(如果是center就是存货量，cell就是功能吧)
            if isinstance(state, list):
                state = list(map(int, state))
            else:
                state = int(state)
            if _key == "cell":
                # f"{i}:\n 状态:{state[1]}\n功能:{state[0]}\n原料:{state[3]}"
                process_states.append(
                    {"id": count, "category": 0, "name": f"工作单元{i}", "material_num": state[3],
                     "state": state[1],
                     "function": state[0]})
            elif _key == "center":
                # f"{i}\n产品:{state}"
                process_states.append({"id": count, "category": 1, "name": f"工作中心{i}", "function": state})
            elif _key == "storage":
                # f"{i}:\n产品:{state[0]}\n数量:{state[1]}"
                process_states.append({"id": count, "category": 2, "name": f"产品中心{i}", "function": state[0],
                                       "product_num": state[1]})
            count += 1
    # 建立边
    for _key, _edge in raw_edge_index.items():
        node1, node2 = _key.split("_to_")
        edge_T = _edge.T.tolist()
        if node1 == "center" and node2 == "storage":
            for __edge in edge_T:
                process_edge.append(
                    {"source": __edge[0] + size[0], "target": __edge[1] + size[0] + size[1]}
                )
        elif node1 == "cell" and node2 == "center":
            for __edge in edge_T:
                process_edge.append({"source": __edge[0], "target": __edge[1] + size[0]})
        elif node1 == "storage" and node2 == "cell":
            for __edge in edge_T:
                process_edge.append({"source": __edge[0] + size[0] + size[1], "target": __edge[1]})
    return {"nodes": process_states, "edges": process_edge}


class EnvRun:
    def __init__(
            self,
            work_center_num,
            fun_per_center,
            function_num,
            device,
            episode_step_max=256,
            product_goal=500,
            product_goal_scale=0.2,
    ):
        self.device = device
        self.edge_index: Dict[str, torch.Tensor] = {}
        self.work_center_num = work_center_num
        self.work_cell_num = self.work_center_num * fun_per_center
        self.func_per_center = fun_per_center
        # 取得三个类型的初始id，要不然edge_index会因为没有对应节点而报错
        # self.center_init_id = WorkCenter.next_id
        # self.work_cell_init_id = WorkCell.next_id
        # self.storage_init_it = StorageCenter.next_id

        self.function_num = function_num
        self.center_num = function_num
        self.center_state_num = 2
        self.read_edge = []
        # 随机生成一个func_per*center_num的矩阵，每列对应一个工作中心
        function_matrix = select_functions(
            0,
            function_num - 1,
            work_center_num,
            fun_per_center,
        )
        # 初始化工作中心
        # TODO 生成一个顺序，然后给workcenter和workcell赋值
        # 计算一下有多少个工作单元，然后生成id列表
        cell_num = function_matrix.size
        cell_id_list = np.arange(cell_num).reshape(-1, function_num)
        # 这个初始化的顺序和工作单元的id顺序也是一致的
        self.work_center_list: list[WorkCenter] = [WorkCenter(i, f, _id, fun_per_center) for i, f, _id in
                                                   zip(range(self.work_center_num), function_matrix, cell_id_list)]

        self.function_group = self.get_function_group()
        # 各级别生产能力，这个应该排除同一个节点拥有两个相同单元
        self.product_capacity = [0 for _ in range(self.function_num)]
        for work_center in self.work_center_list:
            fl = work_center.get_all_funcs()
            for i in range(self.function_num):
                # 函数返回值的是funcid
                # where返回的是tuple，需要提取第一项
                indices: np.ndarray = np.where(fl == i)[0]
                # indices是工作中心中工作单元列表的index，不是id
                # 反正同时只能有一个工作中心工作，所以就取第一个就行了
                if len(indices) > 0:
                    self.product_capacity[i] += work_center.get_cell_speed(int(indices[0]))

        # 根据生产能力和最大步数计算生产目标数量
        # 根据水桶效应，选择最低的生产能力环节代表
        # TODO 每个工步的生产能力其实是波动的，因为其实是工作中心的生产能力
        desire_product_goal = int(
            product_goal_scale * episode_step_max * min(self.product_capacity)
        )
        if abs(product_goal - desire_product_goal) < 100:
            self.product_goal = desire_product_goal
        else:
            self.product_goal = desire_product_goal
        # 初始化集散中心
        self.storage_list: list[StorageCenter] = [StorageCenter(i, self.product_goal, self.function_num) for i in
                                                  range(self.center_num)]
        # 生成物流运输中心代号和中心存储物料的对应关系
        self.center_product: np.ndarray = np.zeros((self.center_num, 2), dtype=int)
        # 这里其实是把第一位是center的id，第二位是product的id
        for i in range(function_num):
            self.center_product[i, 0] = self.storage_list[i].get_id()
            self.center_product[i, 1] = i
        # 产品数量
        self.step_products = np.zeros(function_num)
        # 一次循环的step数量
        self.episode_step = 0
        # 奖励和完成与否
        self.reward = 0
        self.done = 0
        logger.info(f"目标生产数量:{self.product_goal}")

        self.episode_step_max = episode_step_max
        # 初始化边
        self.build_edge()

    @deprecated
    def show_graph(self, step):
        # norm_data: Data = data.to_homogeneous()
        process_label = {}
        process_edge = []
        raw_dict = self.read_state()
        # 这个size是有多少个workcell，主要是为了重新编号
        size = []
        # 总节点数量
        count = 0
        for _key, _value in raw_dict.items():
            size.append(len(_value))
            # 按照顺序一个一个处理成字典和元组形式，state是一行的数据
            for i, state in enumerate(_value):
                # 根据键值不同设置不同的属性
                # cell的function指的是其生成的产品id
                if isinstance(state, list):
                    state = list(map(int, state))
                else:
                    state = int(state)
                if _key == "cell":
                    process_label[
                        count
                    ] = f"{i}:\n 状态:{state[1]}\n功能:{state[0]}\n原料:{state[3]}"
                elif _key == "center":
                    process_label[count] = f"{i}\n产品:{state}"
                elif _key == "storage":
                    process_label[count] = f"{i}:\n产品:{state[0]}\n数量:{state[1]}"
                count += 1
        # 建立边
        for _key, _edge in self.edge_index.items():
            node1, node2 = _key.split("_to_")
            edge_T = _edge.T.tolist()
            if node1 == "center" and node2 == "storage":
                for __edge in edge_T:
                    process_edge.append(
                        (__edge[0] + size[0], __edge[1] + size[0] + size[1])
                    )
            elif node1 == "cell" and node2 == "center":
                for __edge in edge_T:
                    process_edge.append((__edge[0], __edge[1] + size[0]))
            elif node1 == "storage" and node2 == "cell":
                for __edge in edge_T:
                    process_edge.append((__edge[0] + size[0] + size[1], __edge[1]))
        graph = nx.DiGraph()
        graph.add_nodes_from([i for i in range(count)])
        self.read_edge = process_edge
        graph.add_edges_from(process_edge)
        # if count==sum(size):
        nn = [
            [i for i in range(size[0])],
            [i for i in range(size[0], size[0] + size[1])],
            [i for i in range(size[0] + size[1], size[0] + size[1] + size[2])],
        ]
        pos = gen_pos(nn, [i for i in range(count)])
        plt.figure(figsize=(10, 5))
        nx.draw_networkx_nodes(
            graph,
            pos=pos,
            nodelist=[i for i in range(size[0])],
            node_color="white",
            node_size=500,
            edgecolors="black",
            node_shape="s",
            linewidths=1,
        )
        nx.draw_networkx_nodes(
            graph,
            pos,
            nodelist=[i for i in range(size[0], size[0] + size[1])],
            node_color="white",
            node_size=500,
            edgecolors="black",
            node_shape="s",
            linewidths=1,
        )
        nx.draw_networkx_nodes(
            graph,
            pos,
            nodelist=[i for i in range(size[0] + size[1], size[0] + size[1] + size[2])],
            node_color="white",
            node_size=500,
            edgecolors="blue",
            node_shape="s",
            linewidths=1,
        )
        nx.draw_networkx_labels(graph, pos=pos, labels=process_label, font_size=6)
        nx.draw_networkx_edges(
            graph,
            pos,
            edgelist=process_edge,
            edge_color="black",
            arrows=True,
            connectionstyle="arc3,rad=0.12",
            node_size=500,
            node_shape="s",
        )
        # nx.draw(graph, with_labels=True)

        plt.savefig(f"./graph/{step}.png", dpi=500, bbox_inches="tight")
        plt.show()
        plt.close()
        return
        # process_edge = []
        # node = []
        # hetero_data = HeteroData()
        # # 节点信息
        # for key, _value in self.obs_states.items():
        #     hetero_data[key].x = _value
        #     # 边信息
        # for key, _value in self.edge_index.items():
        #     node1, node2 = key.split("_to_")
        #     hetero_data[(f"{node1}", f"{key}", f"{node2}")].edge_index = _value
        # graph: nx.Graph = to_networkx(data=hetero_data, to_undirected=False)
        # for key in self.edge_index.keys():
        #     graph.add_edges_from([self.edge_index[key]])
        # nx.draw(graph, with_labels=True)
        # plt.show()
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
                id_center=self.center_product)
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
            # 这个是取出当前正在工作的，因为work如果在前边的话func改变后product对不上号了
            _funcs: int = work_center.get_func()
            _products = work_center.get_product()
            # 取出所有物品，放入center中
            self.storage_list[_funcs].receive_product(_products)
            # 当前步全部的product数量
            products[_funcs] += _products
            # 转移产品，清空workcell库存
            work_center.send_product()
        self.step_products = products
        # 处理原料
        # 根据是否接收物料的这个动作空间传递原料
        for work_center in self.work_center_list:
            id_funcs = work_center.get_all_cellid_func()
            # 第一列是cellid，第二列是functionid
            # 我就是让他相乘一个系数，如果不分配，这个系数就是0
            # 这里得去掉0功能和最后一个功能
            for _func in id_funcs:
                # 第一位是cellid，第二位是functionid
                assert isinstance(_func, np.ndarray)
                _func_id: int = int(_func[1])
                _cell_id: int = int(_func[0])
                # 如果功能为0就不能运送了，如果功能为最后一个也不能运送走了
                if _func_id != 0:
                    self.storage_list[_func_id - 1].send_product(
                        int(products[_func_id - 1] * ratio[_cell_id])
                    )
            # collect是每个cell的权重
            # 这里注意！！！！，因为funcs要-1才是需要的原料
            material_list = (
                    products[id_funcs[:, 1] - 1] * ratio[id_funcs[:, 0]].tolist()
            )
            material_list = list(map(int, material_list))
            work_center.receive_material(material_list)

    def update_all_work_center(self, workcell_action: np.ndarray):
        for action, work_center in zip(workcell_action, self.work_center_list):
            work_center.work(int(action))

    def update_all(self, all_action: Dict[str, torch.Tensor]):
        # action按节点类别分开

        # 针对workcenter的
        self.deliver_centers_material(all_action["cell"].numpy())
        self.update_all_work_center(all_action["center"].numpy())

        # 额定扣血
        stable_reward = (
                -0.05
                * self.work_cell_num
                / self.function_num
                * max(self.episode_step / self.episode_step_max, 1 / 2)
        )

        # 生产有奖励，根据产品级别加分
        products_reward = 0
        for i, storage in enumerate(self.storage_list):
            # 库存数量/该产品生产能力，当生产最后一个类别的时候不计
            if i == len(self.storage_list):
                break
            products_reward += (
                    -0.002
                    * storage.get_product_num()
                    / self.product_capacity[i]
                # * (i)
                # / self.function_num
            )
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
        # 实现任务目标
        elif self.storage_list[-1].get_product_num() > self.product_goal:
            self.reward += 5
            self.done = 1

    def get_obs(
            self,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], float, int, int]:
        work_cell_state_list = [cell for work_center in self.work_center_list for cell in
                                work_center.get_all_cell_state()]
        # for work_center in self.work_center_list:
        #     a += work_center.get_all_cell_state()
        # 按cellid排序，因为要构造数据结构，其实不用排序，因为本来就是按顺序生成的。。。。
        # sort_state = sorted(a, key=lambda x: x[0])
        work_cell_states = torch.stack(work_cell_state_list).to(self.device)
        # 这里会因为cell_id的自增出问题
        storage_state_list = [storage.get_state() for storage in self.storage_list]
        storage_states = torch.stack(storage_state_list).to(self.device)
        work_center_state_list = [center.get_state() for center in self.work_center_list]
        # for center in self.work_center_list:
        #     work_center_state_list.append(center.get_state())
        work_center_states = torch.stack(work_center_state_list).to(self.device)

        obs_states: Dict[str, torch.Tensor] = {
            "cell": work_cell_states,
            "center": work_center_states,
            "storage": storage_states,
        }
        # 保证obsstates和edgeindex都转到cuda上

        return (
            obs_states,
            self.edge_index,
            self.reward,
            self.done,
            self.episode_step,
        )

    def online_state(self):
        raw_state = self.read_state()
        return process_raw_data(self.edge_index, raw_state)

    def read_state(self):
        # a = []
        # for work_center in self.work_center_list:
        #     a += work_center.read_all_cell_state()
        # b = []
        # for center in self.work_center_list:
        #     b.append(int(center.read_state()))
        # c = []
        # for storage in self.storage_list:
        #     c.append(storage.read_state())
        a = [cell for work_center in self.work_center_list for cell in work_center.read_all_cell_state()]

        b = [int(center.read_state()) for center in self.work_center_list]

        c = [storage.read_state() for storage in self.storage_list]
        return {
            "cell": a,
            "center": b,
            "storage": c,
        }

    def get_function_group(self):
        work_function = [cell for work_center in self.work_center_list for cell in work_center.get_all_cellid_func()]
        # for work_center in self.work_center_list:
        #     work_function += work_center.get_all_cellid_func()
        # 排序,不需要
        # sort_func = sorted(work_function, key=lambda x: x[0])
        work_function = np.array(work_function).T
        # 针对function的分类
        unique_labels = np.unique(work_function[1])
        # 如果直接用where的话，返回的就是符合要求的index，
        # 刚好是按照cellid排序的，所以index就是cellid
        grouped_indices = [
            np.where(work_function[1] == label) for label in unique_labels
        ]
        return grouped_indices

    def reset(self):
        self.step_products = np.zeros(self.function_num)
        self.reward = 0
        self.done = 0
        self.episode_step = 0
        for center in self.work_center_list:
            center.reset_state()
        for center in self.storage_list:
            center.product_num = 0
