from typing import List, Tuple, Union
from BasicClass import BasicClass
from numpy import ndarray, dtype
from model.WorkCell import WorkCell
import numpy as np
import torch
from loguru import logger
from model.StateCode import CenterCode, CellCode
from typing import ClassVar
from torch import Tensor


# 这个类是用来定义加工中心的，一个加工中心包括多个加工单元，但同一时间只能有一个加工单元工作
class WorkCenter(BasicClass):

    def __init__(self, process_id: int, speed_list: Tensor, init_func: int, product_num: int, process_num: int):
        """
        工作中心初始化
        Args:
            process_id: 该工作中心位于第几道工序
            speed_list: 该工作中心所具有的功能的速度列表，例如【10,20,30】单位：半成品数量/单位时间
        """
        super().__init__(process_id)
        # 这个是工作中心属于第几道工序
        self._speed_list = speed_list.to(dtype=torch.int)
        # 这次的list长度不包括nan，有几个就是几个，但是对应序号是【1,2】这样，不是【nan，1,2】
        # 所以在选择working_cell的时候需要注意workcell_list长度不是和funclist中的元素一样长的，可能会出现越界
        # 比如只有两个，但是里边是[1,2]选了2就会导致越界
        self._func_list = torch.arange(speed_list.shape[0], dtype=torch.int)
        # self._func_list: Tensor = torch.nonzero(~torch.isnan(speed_list), as_tuple=False).flatten()
        # 构建workcell
        self.workcell_list: List[WorkCell] = [
            WorkCell(func.item(), self._speed_list[func].item(), process_id, material=0) for func
            in
            self._func_list]
        self._working_status = CenterCode.center_ready
        self._working_func: int = init_func
        self._working_speed = int(self._speed_list[init_func].item())
        self._working_cell = self.workcell_list[init_func]
        self._all_cell_id = torch.tensor([workcell.id for workcell in self.workcell_list], dtype=torch.int)
        self._process_num = process_num if process_num > 1 else 2
        self._product_num = product_num
        self._center_state_code_len = len(CenterCode)

    @property
    def func_list(self):
        return self._func_list

    @property
    def working_status(self):
        return self._working_status

    def build_edge(self, storage_list: list[Tensor]) -> Tuple[Tensor, Tensor, Tensor, Union[Tensor, None], Tensor]:
        """
        创建该工作中心的边信息
        Args:
            storage_list: 产品中心的id列表

        Returns:
            分成三类的边

        """
        # 还是得有center作为节点，因为center需要聚合多个cell和下一级storage的信息得出该节点是否工作
        _cell2center_list = [(cell.id, self.id) for cell in self.workcell_list if cell.speed != 0]
        _storage2center_list = [(_storage_id, self.id) for
                                (_storage_id, _product_id) in storage_list[self.process]]
        # cell通过storage和center的信息判断启动哪个节点
        _center2cell_list = [(self.id, cell.id) for cell in self.workcell_list]
        #                       cell.get_function() == _product_id and self.category == _category_id]
        _storage2cell_tensor = None
        # 如果上一级工序包括这一级的功能，就是这一级的原材料上一级都有
        func_need_match = self._func_list.clone().view(-1, 1)
        _storage2cell_tensor = torch.empty((2, 0), dtype=torch.long)
        # 以storage0原料仓库为起始节点，而不是工作单元0级
        for i, _storage in enumerate(reversed(storage_list[0:self.process])):
            # 看看有哪些重复的，会广播为（func_list_num，storage_num）的tensor，
            # 所以每个点的纵坐标（第1维度）是storage的index，横坐标是func的index
            # 因为storage是行的，需要复制三次变成3*3,每行都一样的，所以第一维度不一样，但是第零维度是一样的
            # 所以得出结果的第0维度是storage的index，第1维度是func的index
            mask = torch.eq(_storage[:, 1], func_need_match)
            # 找到id
            # 剔除-1找到索引
            _storage_indexes = torch.nonzero(mask)[:, 1]
            _cell_indexes = torch.nonzero(mask)[:, 0]
            func_need_match[_cell_indexes] = -1
            _storage_ids = _storage[_storage_indexes, 0]
            _cell_ids = torch.tensor([self.workcell_list[index].id for index in _cell_indexes], dtype=torch.long)
            # 先把两个id堆叠
            tmp = torch.stack((_storage_ids, _cell_ids), dim=0)
            # 然后再循环连接
            _storage2cell_tensor = torch.cat((_storage2cell_tensor, tmp), dim=1)
            if func_need_match.sum() == -1 * len(self._func_list):
                break

        # _storage2cell_list = [(_storage_id, cell.id) for cell in self.workcell_list for
        #                       (_storage_id, _product_id) in storage_list[self.process]]
        # center和storage也需要向storage

        # _center2storage_list = [(self.id, _storage_id) for
        #                         (_storage_id, _product_id) in storage_list if
        #                         self.f == _category_id]

        _cell2storage_list = [(cell.id, _storage_id) for cell in self.workcell_list for
                              (_storage_id, _product_id) in storage_list[self.process] if
                              cell.function == _product_id and cell.speed != 0]

        _cell2storage_tensor: Tensor = torch.tensor(_cell2storage_list, dtype=torch.long).T
        _cell2center_tensor: Tensor = torch.tensor(_cell2center_list, dtype=torch.long).T
        _storage2center_tensor: Tensor = torch.tensor(_storage2center_list, dtype=torch.long).T
        _center2cell_tensor: Tensor = torch.tensor(_center2cell_list, dtype=torch.long).T

        return _cell2center_tensor, _cell2storage_tensor, _storage2center_tensor, _storage2cell_tensor, _center2cell_tensor

    def receive(self, materials: int, func: int):
        """
        接收原材料产品
        Args:
            func: 功能
            materials: 该工作中心接收的当前运行功能的原料
        """
        if self.speed_list[func].item() != 0:
            self.workcell_list[func].receive(materials)
        # 0号工序自动获取加工速度的一批次原料，cell级别已经忽略了，这里不用了
        # if self.process == 0:
        #     self._working_cell.receive(self._working_speed)

    def work(self, activate_cell: int, on_off: int):
        """
        加工
        Args:
            on_off:
            activate_cell: 加工动作，0或需要工作的工作单元在workcell_list中的位置
        """
        # 这里目前改为了工作中心选择哪个工作单元工作
        # 如果是0，就是全部不工作 如果选了没有的功能就直接返回ready
        if on_off == 0 or self.workcell_list[activate_cell].speed == 0:
            self._working_status = CenterCode.center_ready
            for cell in self.workcell_list:
                cell.work(0)
            return
        # 如果全都工作才能走到这里
        state = CellCode.workcell_ready
        for i, cell in enumerate(self.workcell_list):
            if i == activate_cell:
                state = cell.work(1)
                # 如果正常工作则修改work_center的状态
                self._working_cell = cell
                self._working_func = cell.function
                self._working_speed = cell.speed
            else:
                cell.work(0)
        if state == CellCode.workcell_working:
            self._working_status = CenterCode.center_working
        # if self.working_func == 0:
        #     logger.info(
        #         f"{self.id}号工作中心工作状态：{self._working_status},{self._working_func},{self._working_speed},{self._working_cell.product_count}")

    def reset(self):
        for cell in self.workcell_list:
            cell.reset()

    def send(self) -> list[int]:
        return [cell.send() for cell in self.workcell_list]

    @property
    def working_func(self):
        return self._working_func

    @property
    def working_speed(self):
        return self._working_speed

    @property
    def all_cell_id(self) -> Tensor:
        return self._all_cell_id

    def get_func_list(self):
        return self._func_list

    def get_all_cell_status(self, max_speed=1):
        return [cell.status(max_speed, self._product_num, self._process_num) for cell in self.workcell_list]

    def status(self):
        func_norm = self._working_func / (self._product_num - 1)
        process_norm = self.process / self._process_num
        state_norm = self._working_status.value / (self._center_state_code_len - 1)
        return torch.tensor(
            [
                func_norm,
                process_norm,
                state_norm,
            ],
            dtype=torch.float32,
        )

    def read_state(self) -> tuple[int, int, int, int]:

        # total_m = [cell.materials for cell in self.workcell_list]
        # total_p = [cell.product_count for cell in self.workcell_list]
        return self.working_func, self.working_status.value, self.working_speed, self.process

    @property
    def speed_list(self):
        return self._speed_list
