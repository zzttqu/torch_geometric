from typing import List, Tuple, Generator, Any, Union

from numpy import ndarray, dtype
from model.WorkCell import WorkCell
import numpy as np
import torch
from loguru import logger
from model.StateCode import StateCode
from typing import ClassVar


# 这个类是用来定义加工中心的，一个加工中心包括多个加工单元，但同一时间只能有一个加工单元工作
class WorkCenter:
    _next_id: ClassVar = 0

    def __init__(self, category: int, speed_list: torch.Tensor, init_func: int):
        """
        工作中心初始化
        Args:
            category: 该工作中心位于第几道工序
            speed_list: 该工作中心所具有的功能的速度列表，例如【10,20,30】单位：半成品数量/单位时间
        """

        self._id = WorkCenter._next_id
        self.category = category
        # 计数含nan元素个数
        self.func_num = torch.count_nonzero(speed_list).item()
        func_list = torch.arange(self.func_num)
        # 构建workcell
        self.workcell_list: List[WorkCell] = [
            WorkCell(func, self.func_num, speed) if not torch.isnan(speed) else None for
            func, speed in zip(func_list, speed_list)]
        # 这个是工作中心属于第几道工序
        self.category = category
        self.func = init_func
        self.speed = speed_list[init_func]
        self.product = 0
        self.working_cell = self.workcell_list[init_func]
        self.all_cell_id = [workcell.get_id() for workcell in self.workcell_list]

    def build_edge(self, storage_list: Union[torch.Tensor, np.ndarray]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        创建该工作中心的边信息
        Args:
            storage_list: 产品中心的id列表

        Returns:
            分成三类的边

        """
        """
        # 建立一个work-center内部节点的联系
        _edge = []
        # 从cell到center
        for cell in self.workcell_list:
            _edge.append((cell.get_id(), self.get_id()))
        # 如果是高维的，需要单独处理.T不行，必须是.t()

        center_edge: torch.Tensor = torch.tensor(_edge, dtype=torch.long)

        # emmm如果是一维的，两种方法都不行，必须增加一个维度
        if center_edge.dim() == 1:
            center_edge = center_edge.unsqueeze(0).t()
        center_edge = center_edge.t()
        """
        _cell2storage_list = [(cell.get_id(), _storage_id) for cell in self.workcell_list if cell is not None for
                              (_storage_id, _product_id, _category_id) in storage_list if
                              cell.get_function() == _product_id and self.category == _category_id]
        _storage2cell_list = [(_storage_id, cell.get_id()) for cell in self.workcell_list if cell is not None for
                              (_storage_id, _product_id, _category_id) in storage_list if
                              cell.get_function() == _product_id and self.category - 1 == _category_id]

        _cell2storage_tensor = torch.tensor(_cell2storage_list, dtype=torch.long)
        _storage2cell_tensor = torch.tensor(_storage2cell_list, dtype=torch.long)

        return _cell2storage_tensor, _storage2cell_tensor

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

    @classmethod
    def get_next_id(cls):
        current_id = cls._next_id
        cls._next_id += 1
        return current_id

    @classmethod
    def reset_id(cls):
        cls._next_id = 0

    def send_product(self):
        self.working_cell.send_product()
        self.product = 0

    def get_all_cellid_func(self) -> ndarray[Any, dtype[Any]]:
        id_array = [np.array((workcell.get_id(), workcell.get_function())) for workcell in self.workcell_list]
        id_arrays = np.stack(id_array, dtype=int)
        return id_arrays

    def get_id(self):
        return self._id

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
        func_norm = self.get_func() / self.func_num
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
