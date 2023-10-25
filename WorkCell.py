from typing import Union
import numpy as np
import torch

from envClass import StateCode


class WorkCell:
    next_id = 0

    def __init__(self, function_id, work_center_id, speed=6, materials=6, products=0):
        super().__init__()
        # 需要有当前这个工作单元每个功能的备件，每个功能生产效率
        self._id = WorkCell.next_id
        self.work_center_id = work_center_id
        WorkCell.next_id += 1
        self.function = function_id
        self.speed = speed
        self.materials = materials
        self.products = products
        self.health = 100
        self.state = StateCode.workcell_ready

    def recive_material(self, num: int):
        """_summary_

        Args:
            num (int): 接受原料，但是如果功能为0则直接加speed
        """
        if self.function == 0:
            self.materials += self.speed
        else:
            # 或者接收原材料
            self.materials += num

    def send_product(self):
        """
        转移生产出来的产品
        """
        current_product = self.products
        self.products = 0
        return current_product

    def func_err(self):
        """工作中心是否因为错误启动工作单元而报错"""
        self.state = StateCode.workcell_function_error

    def work(self, action: int) -> StateCode:
        """工作单元运行

        Args:
            action (int): 工作单元的动作，只有0：停止和1：运行

        Returns:
            StateCode: 当前工作单元的状态
        """
        # 工作/继续工作，就直接修改状态了，不用重置function_err
        if action == 1:
            self.state = StateCode.workcell_working
        # 停止工作
        elif action == 0:
            self.state = StateCode.workcell_ready
        # 检查当前状态
        self.state_check()
        if self.state == StateCode.workcell_working:
            # 工作中
            # 2是生产库存数量，0是生产速度,1是原料数量
            self.products += self.speed
            self.materials -= self.speed
            # self.health -= 0.1
        return self.state

    def state_check(self):
        """检测目前工作单元状态，检测原料是否够生产"""
        # 低健康度
        # if self.health < 50:
        #      = None
        #     self.state = StateCode.workcell_low_health
        # 缺少原料
        if self.materials < self.speed:
            self.state = StateCode.workcell_low_material
        # 不缺货就变为ready状态
        elif (
            self.materials >= self.speed
            and self.state == StateCode.workcell_low_material
        ):
            self.state = StateCode.workcell_ready

    def reset_state(self):
        self.state = StateCode.workcell_ready
        # 给一个基础的原料
        self.materials = self.speed
        self.products = 0

    # 状态空间
    def get_state(self) -> torch.Tensor:
        """获取工作单元状态

        Returns:
            torch.Tensor: 第一项是工作单元id
            第二项是工作单元隶属的工作中心id
            第三项是当前的功能
            第四项是一个工步的生产能力
            第五项是当前的产品数量
        """
        # TODO  改为onehot编码的function和state
        return torch.tensor(
            [
                self.function,
                self.state.value,
                # self.work_center_id,
                self.speed,
                # self.products,
                self.materials
            ],
            dtype=torch.float32,
        )

    # 功能id
    def get_function(self):
        return self.function

    def get_id(self):
        return self._id

    def get_speed(self):
        return self.speed

    def get_products(self):
        return self.products

    def get_materials(self):
        return self.materials
