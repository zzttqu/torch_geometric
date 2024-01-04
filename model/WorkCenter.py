from typing import List, Tuple, Union

from numpy import ndarray, dtype
from model.WorkCell import WorkCell
import numpy as np
import torch
from loguru import logger
from model.StateCode import StateCode
from typing import ClassVar
from torch import Tensor


# 这个类是用来定义加工中心的，一个加工中心包括多个加工单元，但同一时间只能有一个加工单元工作
class WorkCenter:
    _next_id: ClassVar = 0

    def __init__(self, category: int, speed_list: Tensor, init_func: int):
        """
        工作中心初始化
        Args:
            category: 该工作中心位于第几道工序
            speed_list: 该工作中心所具有的功能的速度列表，例如【10,20,30】单位：半成品数量/单位时间
        """

        self._id = WorkCenter.get_next_id()
        # 这个是工作中心属于第几道工序
        self._category = category
        self._speed_list = speed_list
        # 计数含nan元素个数
        self.func_num = torch.count_nonzero(speed_list).item()
        # 这次的list长度不包括nan，有几个就是几个，但是对应序号是【1,2】这样，不是【nan，1,2】
        self.func_list = torch.nonzero(~torch.isnan(speed_list), as_tuple=False).flatten()
        # 构建workcell
        self.workcell_list: List[WorkCell] = [
            WorkCell(func, self.func_num, speed) for func, speed in zip(self.func_list, speed_list) if
            not torch.isnan(speed)
        ]

        self.working_func = init_func
        self.working_cell = self.workcell_list[init_func]
        self.all_cell_id = Tensor([workcell.id for workcell in self.workcell_list])

    @property
    def id(self) -> int:
        return self._id

    @property
    def category(self) -> int:
        return self._category

    @property
    def speed_list(self) -> Tensor:
        return self._speed_list

    def build_edge(self, storage_list: Union[Tensor, np.ndarray]) -> Tuple[
        Tensor, Tensor, Tensor, Tensor]:
        """
        创建该工作中心的边信息
        Args:
            storage_list: 产品中心的id列表

        Returns:
            分成三类的边

        """
        # 还是得有center作为节点，因为center需要聚合多个cell和下一级storage的信息得出该节点是否工作
        _cell2center_list = [(cell.id, self.id) for cell in self.workcell_list]
        _storage2center_list = [(_storage_id, self.id) for
                                (_storage_id, _product_id, _category_id) in storage_list if
                                self.category == _category_id]
        # cell通过storage和center的信息判断启动哪个节点
        _center2cell_list = [(self.id, cell.id) for cell in self.workcell_list]
        #                       cell.get_function() == _product_id and self.category == _category_id]
        _storage2cell_list = [(_storage_id, cell.id) for cell in self.workcell_list for
                              (_storage_id, _product_id, _category_id) in storage_list if
                              cell.get_function() == _product_id and self.category - 1 == _category_id]
        # _center2storage_list = [(self.id, _storage_id) for
        #                         (_storage_id, _product_id, _category_id) in storage_list if
        #                         self.category == _category_id]

        # _cell2storage_list = [(cell.id, _storage_id) for cell in self.workcell_list if cell is not None for
        #                       (_storage_id, _product_id, _category_id) in storage_list if

        _cell2center_tensor = Tensor(_cell2center_list, dtype=torch.long)
        _storage2center_tensor = Tensor(_storage2center_list, dtype=torch.long)
        _storage2cell_tensor = Tensor(_storage2cell_list, dtype=torch.long)
        _center2cell_tensor = Tensor(_center2cell_list, dtype=torch.long)

        return _cell2center_tensor, _storage2center_tensor, _storage2cell_tensor, _center2cell_tensor

    def receive_material(self, materials: int):
        """
        接收原材料产品
        Args:
            materials: 该工作中心接收的当前运行功能的原料
        """
        self.workcell_list[self.working_func].receive_material(materials)
        # 0号工序自动获取加工速度的一批次原料
        if self.category == 0:
            self.workcell_list[self.working_func].receive_material(self.speed_list[self.working_func])

    def work(self, activate_cell: int, on_off: int):
        """
        加工
        Args:
            on_off: 
            activate_cell: 加工动作，0或需要工作的工作单元在workcell_list中的位置
        """
        # 这里目前改为了工作中心选择哪个工作单元工作
        # 如果是0，就是全部不工作
        if on_off == 0:
            for cell in self.workcell_list:
                cell.work(0)
        else:
            for i, cell in enumerate(self.workcell_list):
                if i == activate_cell:
                    state = cell.work(1)
                    # 如果正常工作则修改work_center的状态
                    if state == StateCode.workcell_working:
                        self.working_cell = cell
                        self.working_func = cell.get_function()
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

    def get_process(self):
        return self.category

    def get_func(self):
        return self.working_func

    def send_product(self) -> int:
        return self.workcell_list[self.working_func].send_product()

    def get_id(self):
        return self.id

    def get_all_cell_id(self) -> Tensor:
        return self.all_cell_id

    def get_all_cell_state(self):
        return [cell.get_state() for cell in self.workcell_list]

    def get_func_list(self):
        return self.func_list

    def reset_state(self):
        for cell in self.workcell_list:
            cell.reset_state()
        return

    def get_state(self):
        func_norm = self.get_func_list() / self.func_num
        return Tensor(
            [
                func_norm,
            ],
            dtype=torch.float32,
        )

    def read_state(self):
        return self.get_func_list()

    def read_all_cell_state(self):
        a = []
        for cell in self.workcell_list:
            a.append(cell.read_state())
        return a
