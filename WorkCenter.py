from typing import List, Union, Tuple

import torch
from torch import Tensor
import numpy as np


# 这个类是用来定义加工中心的，一个加工中心包括多个加工单元，但同一时间只能有一个加工单元工作
class WorkCenter:
    next_id = 0

    def __init__(self, function_list: np.ndarray) -> None:
        from WorkCell import WorkCell

        self.id = WorkCenter.next_id
        WorkCenter.next_id += 1
        self.workcell_list: List[WorkCell] = []
        self.function_list: np.ndarray = function_list

        # 构建workcell
        for f in function_list:
            self.workcell_list.append(WorkCell(f, self.id))
        self.func = self.workcell_list[0].get_function()
        self.product = 0
        self.working_cell = self.workcell_list[0]
        self.all_cell_id = [workcell.get_id() for workcell in self.workcell_list]

    def build_edge(self, id_center) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
        # 如果为int说明这个center只有一个0号节点，也就直接给0功能节点加数值就行了

        # materials如果是0号就只是int，需要判断
        # 这个material是全部的
        # 这个应该可以改成类似查表的，用cellid直接查在list中的位置
        for cell, material in zip(self.workcell_list, materials):
            cell.recive_material(material)

    def move_product(self, products_list: List[int]):
        for cell in self.workcell_list:
            products_list[cell.get_function()] = cell.send_product()

    def work(self, actions: np.ndarray):
        # 如果同时工作的单元数量大于1，就会报错，惩罚就是当前步无法工作
        if np.sum(actions == 1) > 1:
            for cell in self.workcell_list:
                cell.func_err()
        # 如果正常就正常
        else:
            for cell, action in zip(self.workcell_list, actions):
                state = cell.work(action)
                # 表示当前工作单元的功能
                from envClass import StateCode

                if state == StateCode.workcell_working:
                    self.working_cell = cell
                    self.func = cell.get_function()
                    self.product = cell.get_products()

    def send_product(self):
        self.working_cell.send_product()

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

    def get_cell_speed(self, indexs: Union[np.ndarray, int]) -> int:
        if isinstance(indexs, int):
            speed = self.workcell_list[indexs].get_speed()
        else:
            # 输入是cell位置
            speed_list = [self.workcell_list[index].get_speed() for index in indexs]
            speed = sum(speed_list)
        return speed

    def get_all_cell_state(self):
        return [cell.get_state() for cell in self.workcell_list]

    def get_func(self):
        return self.func

    def get_product(self):
        return self.product
