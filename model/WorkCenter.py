from typing import List, Tuple
from model.WorkCell import WorkCell
import numpy as np
import torch
from loguru import logger
from model.StateCode import StateCode
from typing import ClassVar


# 这个类是用来定义加工中心的，一个加工中心包括多个加工单元，但同一时间只能有一个加工单元工作
class WorkCenter:
    next_id: ClassVar = 0

    def __init__(self, function_list: np.ndarray, max_func_num):
        """
        工作中心初始化
        Args:
            function_list: 该工作中心所具有的功能列表，例如【0,1】
            max_func_num: 最大功能数，用于规范化
        """

        self.id = WorkCenter.next_id
        self.max_func_num = max_func_num
        WorkCenter.next_id += 1
        self.workcell_list: List[WorkCell] = []
        self.function_list: np.ndarray = function_list

        # 构建workcell
        for f in function_list:
            self.workcell_list.append(WorkCell(f, self.id, self.max_func_num))
        self.func = self.workcell_list[0].get_function()
        self.speed = self.workcell_list[0].get_speed()
        self.product = 0
        self.working_cell = self.workcell_list[0]
        self.all_cell_id = [workcell.get_id() for workcell in self.workcell_list]

    def build_edge(self, id_center) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        创建该工作中心的边信息
        Args:
            id_center: 产品中心的id列表

        Returns:
            分成三类的边

        """
        # 建立一个workcenter内部节点的联系
        _edge = []
        # 从cell到center
        for cell in self.workcell_list:
            _edge.append((cell.get_id(), self.id))
        center_edge = torch.tensor(_edge, dtype=torch.long).squeeze().T
        # 从center到storage
        _edge = []
        _edge1 = []
        # 建立上下游联系
        for cell in self.workcell_list:
            for _center in id_center:
                # _center是一个长度为2的数组，第一位是center的id，第二位是product的id
                # 从center到storage
                if cell.get_function() == _center[1]:
                    _edge.append((self.get_id(), _center[0]))
                # 从storage到center
                elif cell.get_function() - 1 == _center[1]:
                    _edge1.append((_center[0], cell.get_id()))
        product_edge = torch.tensor(_edge, dtype=torch.long).squeeze().T
        if len(_edge1) > 0:
            material_edge = torch.tensor(_edge1, dtype=torch.long).T
        else:
            material_edge = None
        return center_edge, product_edge, material_edge

    def receive_material(self, materials: List[int]):
        """
        接收原材料产品
        Args:
            materials: 该工作中心接收的原材料产品列表
        """

        # 这个material是全部的
        # 这个应该可以改成类似查表的，用cellid直接查在list中的位置

        for cell, material in zip(self.workcell_list, materials):
            assert isinstance(cell, WorkCell)
            cell.receive_material(material)

    # def move_product(self, products_list: List[int]):
    #     for cell in self.workcell_list:
    #         products_list[cell.get_function()] = cell.send_product()

    def work(self, action: int):
        """
        加工
        Args:
            action: 加工动作，0或需要工作的工作单元在workcell_list中的位置
        """
        # 这里目前改为了工作中心选择哪个工作单元工作
        # 如果是0，就是全部
        if action == 0:
            for cell in self.workcell_list:
                cell.work(0)
        else:
            for i, cell in enumerate(self.workcell_list):
                if i == action - 1:
                    state = cell.work(1)
                    # 如果正常工作则修改work_center的状态
                    if state == StateCode.workcell_working:
                        self.working_cell = cell
                        self.func = cell.get_function()
                        self.speed = cell.get_speed()
                        self.product = cell.get_products()
                else:
                    cell.work(0)

    def send_product(self):
        self.working_cell.send_product()
        self.product = 0

    def get_all_cellid_func(self) -> List:
        a = []
        for workcell in self.workcell_list:
            a.append([workcell.get_id(), workcell.get_function()])
        return a

    def get_all_funcs(self) -> np.ndarray:
        return self.function_list

    def get_id(self):
        return self.id

    def get_all_cell_id(self) -> List[int]:
        return self.all_cell_id

    def get_cell_speed(self, indexes: int) -> int:
        speed = self.workcell_list[indexes].get_speed()
        return speed

    def get_all_cell_state(self):
        return [cell.get_state() for cell in self.workcell_list]

    def get_speed(self):
        return self.speed

    def get_func(self):
        return self.func

    def get_product(self):
        return self.product

    def reset_state(self):
        for cell in self.workcell_list:
            cell.reset_state()
        return

    def get_state(self):
        func_norm = self.get_func() / self.max_func_num
        return torch.tensor(
            [
                func_norm,
            ],
            dtype=torch.float32,
        )

    def read_state(self):
        return self.get_func()

    def read_all_cell_state(self):
        a = []
        for cell in self.workcell_list:
            a.append(cell.read_state())
        return a
