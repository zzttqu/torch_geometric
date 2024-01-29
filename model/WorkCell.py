import numpy as np
import torch
from model.BasicClass import BasicClass
from model.StateCode import CellCode
from loguru import logger


class WorkCell(BasicClass):
    """
    WorkCell 工作单元作为虚构的最小的可重构单位，只能执行一种功能
    """

    def __init__(
            self,
            function_id: int,
            speed: int,
            process_id,
            material: int = 0,
    ):

        """
        WorkCell 初始化函数
        Args:
            function_id:功能id，对应产品，也对应其所用时间
            speed:生产速度，单位时间/每个半成品
        """
        # 需要有当前这个工作单元每个功能的备件，每个功能生产效率
        super().__init__(process_id)
        self._materials = material
        self._init_materials = material
        self._product_count = 0
        self._health = 100
        self._speed = int(speed)
        self._function = function_id
        self.state = CellCode.workcell_ready
        self._cell_state_code_len = len(CellCode)

    @property
    def product_count(self):
        return self._product_count

    @property
    def function(self):
        return self._function

    @property
    def speed(self):
        return self._speed

    def receive(self, num: int):
        """
        接收原料数量
        Args:
            num (int): 接受原料，但是如果功能为0则忽略
        """
        # 接收原材料
        self._materials += num

    def send(self) -> int:
        """
        转移生产出来的产品数量
        Returns:
            当前节拍生产的产品数量
        """
        current_product = self._product_count
        self._product_count = 0
        return int(current_product)

    def func_err(self):
        """
        工作单元报错
        """
        self.state = CellCode.workcell_function_error

    def work(self, action: int) -> CellCode:
        """工作单元运行

        Args:
            action (int): 工作单元的动作，只有0：停止和1：运行

        Returns:
            CellCode: 当前工作单元的状态
        """
        # 工作/继续工作，就直接修改状态了，不用重置function_err
        if action == 1:
            self.state = CellCode.workcell_working
        # 停止工作
        elif action == 0:
            self.state = CellCode.workcell_ready
        # 检查当前状态
        self.state_check()
        if self.state == CellCode.workcell_working:
            # 工作中
            self._product_count += self.speed
            # 如果是0号功能，那就不扣原料
            self._materials -= self.speed
            # self._health -= 0.1
        return self.state

    def state_check(self):
        """
        检测目前工作单元状态，检测原料是否够生产
        """
        # 低健康度
        # if self._health < 50:
        #      = None
        #     self.state = StateCode.workcell_low_health
        # 缺少原料
        if self._materials < self.speed:
            self.state = CellCode.workcell_low_material
        # 不缺货就变为ready状态
        elif (
                self._materials >= self.speed
                and self.state == CellCode.workcell_low_material
        ):
            self.state = CellCode.workcell_ready

    def reset(self):
        """
        重置工作单元状态

        """
        self.state = CellCode.workcell_ready
        # 给一个基础的原料
        self._materials = self._init_materials
        self._product_count = 0

    # 状态空间
    def status(self, max_speed=1, func_num=1, process_num=1) -> torch.Tensor:
        """
        获取工作单元状态，规范化后的
        Returns:
            torch.Tensor: 第一项是工作单元id
            第二项是工作单元隶属的工作中心id
            第三项是当前的功能
            第四项是一个工步的生产能力
            第五项是当前的产品数量
        """
        # 规范化speed和materials
        speed_norm = self.speed / max_speed
        func_norm = self.function / (func_num - 1)
        process_norm = self.process / process_num
        state_norm = self.state.value / (self._cell_state_code_len - 1)
        return torch.tensor(
            [
                self.id,
                func_norm,
                process_norm,
                state_norm,
                speed_norm,
            ],
            dtype=torch.float32,
        )

    def read_state(self) -> list[int]:
        """
        读取可读状态
        Returns:
            可读状态
        """
        return [int(self.function), int(self.state.value), int(self.speed), int(self._materials)]

    @property
    def materials(self):
        return self._materials
