from typing import List
from WorkCell import WorkCell
import torch
from torch import Tensor
import numpy as np


# 这个类是用来定义加工中心的，一个加工中心包括多个加工单元，但同一时间只能有一个加工单元工作
class WorkCenter:
    next_id = 0

    def __init__(self, function_list: np.ndarray) -> None:
        self.id = WorkCenter.next_id
        WorkCenter.next_id += 1
        self.workcell_list: List[WorkCell] = []
        self.function_list = function_list
        # 构建workcell
        for f in function_list:
            self.workcell_list.append(WorkCell(f, self.id))

    def build_edge(self, id_center) -> np.ndarray:
        # 建立内部节点的联系
        _id_list = []
        _edge = []
        for cell in self.workcell_list:
            _id_list.append(cell.get_id())
        for i in _id_list:
            for j in _id_list:
                if i != j:
                    _edge.append((i, j))
        # 建立上下游联系
        # TODO 这里需要根据运输中心的id解决上下游和内部节点的联系
        for cell in self.workcell_list:
            for _center in id_center:
                #_center是一个长度为2的数组，第一位是center的id，第二位是product的id
                if cell.get_function() == _center[1]:

        return np.array(_edge).T

    def get_material(self, materials: List[int]):
        # 这个material是全部的
        for cell in self.workcell_list:
            cell.get_material(materials[cell.get_function()])

    def move_product(self, products_list: List[int]):
        for cell in self.workcell_list:
            products_list[cell.get_function] = cell.move_product()

    def work(self, actions):
        for cell in self.workcell_list:
            cell.work(actions[cell.get_id()])

    def get_function(self):
        return self.function_list

    def get_id(self):
        return self.id

    def get_all_cell_id(self):
        return [workcell.get_id for workcell in self.workcell_list]

    def get_cell_speed(self, index: int):
        return self.workcell_list[index].get_speed()
