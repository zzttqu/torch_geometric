from typing import List, Union

import torch
from torch import Tensor
import numpy as np

from envClass import StateCode


# 这个类是用来定义加工中心的，一个加工中心包括多个加工单元，但同一时间只能有一个加工单元工作
class WorkCenter:
    next_id = 0

    def __init__(self, function_list: np.ndarray) -> None:
        from WorkCell import WorkCell

        self.id = WorkCenter.next_id
        WorkCenter.next_id += 1
        self.workcell_list: List[WorkCell] = []
        self.function_list: List[int] = function_list.tolist()

        # 构建workcell
        for f in function_list:
            self.workcell_list.append(WorkCell(f, self.id))
        self.func = self.workcell_list[0].get_function()
        self.product = 0

    def build_edge(self, id_center) -> Union[torch.Tensor, torch.Tensor]:
        # 建立一个workcenter内部节点的联系
        _id_list = []
        _edge = []
        for cell in self.workcell_list:
            _id_list.append(cell.get_id())
        for i in _id_list:
            for j in _id_list:
                if i != j:
                    _edge.append((i, j))
        center_edge = torch.tensor(np.array(_edge).T, dtype=torch.long)
        _edge = []
        _edge1 = []
        # 建立上下游联系
        for cell in self.workcell_list:
            for _center in id_center:
                # _center是一个长度为2的数组，第一位是center的id，第二位是product的id
                if cell.get_function() == _center[1]:
                    _edge.append((cell.get_id(), _center[0]))
                elif cell.get_function() - 1 == _center[1]:
                    _edge1.append((_center[0], cell.get_id()))
        product_edge = torch.tensor(np.array(_edge).T, dtype=torch.long)
        material_edge = torch.tensor(np.array(_edge1).T, dtype=torch.long)
        return center_edge, product_edge, material_edge

    def recive_material(self, materials: List[int]):
        material_list = []
        # 这个material是全部的
        for cell in self.workcell_list:
            cell.recive_material(materials[cell.get_function()])

    def move_product(self, products_list: List[int]):
        for cell in self.workcell_list:
            products_list[cell.get_function] = cell.move_product()

    def work(self, actions: np.ndarray):
        # 如果同时工作的单元数量大于1，就会报错，惩罚就是当前步无法工作
        if np.sum(actions == 1) > 1:
            for cell in self.workcell_list:
                cell.func_err()
        # 如果正常就正常
        else:
            for i, cell in enumerate(self.workcell_list):
                state = cell.work(actions[i])
                # 表示当前工作单元的功能
                if state == StateCode.workcell_working:
                    self.func = cell.get_function()
                    self.product = cell.get_products()

    def get_all_cell_func(self) -> List:
        a = []
        for workcell in self.workcell_list:
            a.append([workcell.get_id(), workcell.get_function()])
        return a

    def get_id(self):
        return self.id

    def get_all_cell_id(self):
        return [workcell.get_id for workcell in self.workcell_list]

    def get_cell_speed(self, indexs: List[int]) -> int:
        cell_list = [self.workcell_list[index].get_speed() for index in indexs]
        speed = sum(cell_list)
        return speed

    def get_all_cell_state(self):
        return [cell.get_state() for cell in self.workcell_list]

    def get_func(self):
        return self.func

    def get_product(self):
        return self.product
