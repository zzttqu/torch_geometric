from typing import List, Dict, Tuple

from torch_geometric.data import HeteroData
from torch_geometric.utils import to_networkx
import networkx as nx
import numpy as np
import torch
from torch import Tensor

from model.StateCode import StateCode
from model.WorkCell import WorkCell
from model.WorkCenter import WorkCenter
from model.StorageCenter import StorageCenter
from loguru import logger
from torch.distributions import Categorical

# from model.WorkCell import WorkCell
# plt.rcParams["font.sans-serif"] = ["SimHei"]  # 显示中文标签
# plt.rcParams["axes.unicode_minus"] = False
import warnings


def deprecated(func):
    def wrapper(*args, **kwargs):
        warnings.warn("Call to deprecated function.", category=DeprecationWarning)
        return func(*args, **kwargs)

    return wrapper


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
            work_center_init_func,
            order,
            speed_list,
            device,
            episode_step_max=256,
            product_goal=500,
            product_goal_scale=0.2,
    ):
        self.device = device
        self.edge_index: Dict[str, torch.Tensor] = {}
        # 产品订单
        order = torch.tensor([100, 600, 200])
        # 这里应该是对各个工作单元进行配置了
        work_center_init_func = torch.tensor([[3, 3, 10],
                                              [2, 2, 6],
                                              [4, 5, 0],
                                              [3, 0, 12],
                                              [2, 3, 5]])

        speed_list = torch.tensor([[5, 10, 15, 20, 12], [8, 12, 18, torch.nan, 12], [3, 6, torch.nan, 10, 8]]).T
        self.speed_list = speed_list
        self.process_num = work_center_init_func.shape[0]
        self.func_num = order.shape[0]
        self.product_count = torch.zeros(size=(self.process_num, self.func_num), dtype=torch.int)
        self.total_center_num = work_center_init_func.sum()
        # 每个工序的工作中心数量
        self.center_per_process = torch.sum(work_center_init_func, dim=1)
        self.per_process_num = torch.sum(~torch.isnan(speed_list), dim=1)
        self.max_speed = torch.max(speed_list[~torch.isnan(speed_list)])
        self.state_code_size = len(StateCode)
        self.work_center_list: list[WorkCenter] = []
        # 第一层解析工序，第二层解析每个工序中的产品，第三层生成工作中心
        for process, funcs in enumerate(work_center_init_func):
            for func, num in enumerate(funcs):
                self.work_center_list.extend([
                    WorkCenter(process, speed_list[process], func)
                    for _ in range(0, num)
                ])
        # TODO 加工能力代码

        # 初始化货架
        # 货架数量是产品工序和产品类别共同构成的
        # 不包括没有那道工序的半成品
        # 这个是找到不为0的元素位置，是一个（2，n）的tensor
        storage_need_tensor = torch.nonzero(~speed_list.isnan(), as_tuple=False)
        # 根据speed构建storage
        self.storage_list = [StorageCenter(product.item(), process.item(), order[product.item()], self.func_num) for
                             process, product in
                             storage_need_tensor]
        self.storage_id_relation = torch.tensor(
            [(storage.id, storage.product_id, storage.process) for storage in self.storage_list],
            dtype=torch.int)
        self.storage_id_relation: list[Tensor] = [torch.empty((0, 2), dtype=torch.int) for _ in range(self.process_num)]
        # 生成货架和半成品的对应关系
        for storage in self.storage_list:
            storage_data = torch.tensor((storage.id, storage.product_id)).view(1, -1)
            self.storage_id_relation[storage.process] = torch.cat(
                (self.storage_id_relation[storage.process], storage_data), dim=0)

        # 一次循环的step数量
        self.episode_step = 0
        # 奖励和完成与否
        self.reward = 0
        self.done = 0

        self.episode_step_max = episode_step_max
        # 初始化边
        self.build_edge()

    def build_edge(self) -> Dict[str, torch.Tensor]:
        edge_names = ["cell2center", "storage2center", "storage2cell", "center2cell"]
        # logger.info(self.storage_id_relation)
        for edge_name in edge_names:
            self.edge_index[edge_name] = torch.zeros((2, 0), dtype=torch.long)

        for work_center in self.work_center_list:
            # 直接挑出工序对应的储存中心
            edges = work_center.build_edge(storage_list=self.storage_id_relation)

            for edge_name, edge_data in zip(edge_names, edges):
                if edge_data is not None:
                    self.edge_index[edge_name] = torch.cat([self.edge_index[edge_name], edge_data], dim=1)

        for edge_key in self.edge_index.keys():
            self.edge_index[edge_key] = self.edge_index[edge_key].to(self.device)
        return self.edge_index

    def update(self, all_action: dict[str, torch.Tensor]):
        cell_logits = all_action["cell"]
        center_logits = all_action["center"]
        center_ratio = torch.zeros(self.total_center_num, dtype=torch.float32)
        # self.work_center_process中记录了各个process的workcenter数量
        center_id = 0
        # 首先生产
        for process, center_num in enumerate(self.center_per_process):
            # 为了保证不影响softmax，如果是0会导致最后概率不为0，根据每个工序包括的功能数量初始化
            assert isinstance(center_num, int), "center_num 必须是 int"
            _ratio = torch.full((self.per_process_num[process], center_num), -torch.inf)
            for process_index, work_center in enumerate(self.work_center_list[center_id:center_id + center_num]):
                cell_slice = cell_logits[work_center.all_cell_id]
                center_slice = center_logits[work_center.id]
                center_dist = Categorical(logits=center_slice)
                cell_dist = Categorical(logits=cell_slice[:, 0])
                # 这里其实不影响，因为实际上是修改的workcell，但是如果这个工作中心没有这个功能，其实工作单元也没有，那么这个输出的index其实就是celllist的index
                activate_func = cell_dist.sample().item()
                # 这里是放置当前工序各个product的分配率
                _ratio[activate_func, process_index] = cell_slice[activate_func, 1]
                on_or_off = center_dist.sample().item()
                work_center.work(activate_func, on_or_off)
            # 还需要考虑如果所有的cell都选了一个func就会导致全是inf，softmax后的tensor为nan
            _ratio = torch.softmax(_ratio, dim=1)
            # 如果有nan就改成0
            _ratio_without_nan = torch.where(torch.isnan(_ratio), torch.tensor(0.0), _ratio)
            # 因为center不能同时出现在两个工序中，所以_ratio必定同一列只有一个非0的，所以累加起来就变成center_num长度的数组了，就可以给center赋值了，
            # 然后交给之后的物料转运
            ratios = torch.sum(_ratio_without_nan, dim=0)
            center_ratio[center_id:center_id + center_num] = ratios
            # 更新center的id，方便循环内部使用
            center_id += center_num
        # 其次运输
        for storage in self.storage_list:
            process, func1 = storage.process, storage.product_id
            for center in self.work_center_list:
                process, func2 = center.process, center.working_func
                if process == 0:
                    center.receive(1)
                elif process - 1 == process and func1 == func2:
                    materials = storage.send(center_ratio[center.id])
                    center.receive(materials)
                elif process == process and func1 == func2:
                    product = center.send()
                    storage.receive(product)
        # 最后计算奖励
        # self.get_reward()

    def get_reward(self):
        # 额定扣血
        # 暂时删去
        """stable_reward = (
                -0.05
                * self.work_cell_num
                / self.product_num
                * max(self.episode_step / self.episode_step_max, 1 / 2)
        )"""
        # 生产有奖励，根据产品级别加分
        products_reward = 0

        for storage in self.storage_list:
            if storage.product_count > 2 * self.speed_list[storage.process][storage.product_id]:
                products_reward += storage.product_count * 0.01
            self.product_count[storage.process][storage.product_id] = storage.product_count
        for i, storage in enumerate(self.storage_list[:-1]):
            # 只有库存过大的时候才会扣血
            if storage.product_count > self.speed_list[i] * 2:
                # logger.info(f"{i}号库存{storage.get_product_num()}，生产能力{self.product_capacity[i]}")
                products_reward += (
                        -0.01
                        * storage.get_product_num()
                    # * (i)
                    # / self.function_num
                )
        # 最终产物肯定要大大滴加分
        products_reward += 0.05 * self.step_products[-1]
        # 最终产物奖励，要保证这个产物奖励小于扣血
        # goal_reward = self.storage_list[-1].get_product_num() * 0.001
        # self.reward += stable_reward
        # self.reward += goal_reward
        self.reward += products_reward
        self.done = 0
        self.episode_step += 1
        # 相等的时候刚好是episode，就是1,2,3,4，4如果是max，那等于4的时候就应该跳出了
        if self.episode_step >= self.episode_step_max:
            # self.reward -= 5
            self.done = 1
        # 实现任务目标
        elif self.storage_list[-1].get_product_num() > self.product_goal:
            self.reward += 0.5
            self.done = 1

    def get_obs(
            self,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], float, int, int]:
        # 先从workcenter中取出来，再从celllist的list中取出来每个cell的tensor
        work_cell_states = torch.stack([cell for work_center in self.work_center_list for cell in
                                        work_center.get_all_cell_state(self.max_speed, self.func_num,
                                                                       self.state_code_size)], dim=0).to(self.device)
        # for work_center in self.work_center_list:
        #     a += work_center.get_all_cell_state()
        # 按cellid排序，因为要构造数据结构，其实不用排序，因为本来就是按顺序生成的。。。。
        # sort_state = sorted(a, key=lambda x: x[0])
        # 这里会因为cell_id的自增出问题
        storage_states = torch.stack([storage.status() for storage in self.storage_list], dim=0).to(self.device)
        work_center_states = torch.stack([center.status(self.func_num) for center in self.work_center_list], dim=0).to(
            self.device)

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
        a = [cell for work_center in self.work_center_list for cell in work_center.read_all_cell_state()]

        b = [int(center.read_state()) for center in self.work_center_list]

        c = [storage.read_state() for storage in self.storage_list]
        return {
            "cell": a,
            "center": b,
            "storage": c,
        }

    def reset(self):
        self.step_products = np.zeros(self.func_num)
        # 只有重新生成的时候再resetid
        # WorkCenter.reset_id()
        # WorkCell.reset_id()
        # StorageCenter.reset_id()
        self.reward = 0
        self.done = 0
        self.episode_step = 0
        for center in self.work_center_list:
            center.reset()
        for center in self.storage_list:
            center._product_count = 0

    def reinit(self):
        self.reset()
        WorkCenter.reset_class_id()
        WorkCell.reset_class_id()
        StorageCenter.reset_class_id()

    """    @deprecated
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
        products = np.zeros(self.product_num)
        # 处理产品
        for work_center in self.work_center_list:
            # 这个是取出当前正在工作的，因为work如果在前边的话func改变后product对不上号了
            _funcs: int = work_center.get_func_list()
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

    @deprecated
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
        #     graph.add_node(node.id, working=node.get_function())
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
        #             graph.add_edge(work_cell.id, center.cell_id + self.work_cell_num)
        #         # 从中转到下一步
        #         if product_id == cell_fun_id - 1:
        #             # 可视化节点需要id不能重复的
        #             graph.add_edge(center.cell_id + self.work_cell_num, work_cell.id)
        """


if __name__ == '__main__':
    order = torch.tensor([100, 600, 200])
    # 这里应该是对各个工作单元进行配置了
    work_center_init_func = torch.tensor([[3, 3, 10],
                                          [2, 2, 6],
                                          [4, 5, 0],
                                          [3, 0, 12],
                                          [2, 3, 5]])

    speed_list = torch.tensor([[5, 10, 15, 20, 12], [8, 12, 18, torch.nan, 12], [3, 6, torch.nan, 10, 8]]).T
    env = EnvRun(work_center_init_func, order, speed_list, torch.device('cuda:0'))
    logger.info(env.get_obs())
    hh = HeteroData()
    _id = 0
    for key, value in env.get_obs()[0].items():
        _id += value.shape[0]
        logger.info(_id)
        hh[key].x = value
    for key, value in env.get_obs()[1].items():
        a, b = key.split('2')
        hh[a, 'to', b].edge_index = value
    G = to_networkx(hh)
    import matplotlib.pyplot as plt

    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['figure.figsize'] = (15, 15)
    pos = nx.spring_layout(G, k=0.2, scale=0.5)
    pos1 = nx.circular_layout(G)
    pos2 = nx.shell_layout(G)
    node_size = [10 * G.degree(node) for node in G.nodes]
    nx.draw(G, pos2, with_labels=True, node_size=node_size, font_size=5, edge_color='gray', width=1.0, alpha=0.7)

    plt.show()


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
