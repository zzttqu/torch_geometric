
import numpy as np
import torch

from envClass import StateCode


class WorkCell:
    next_id = 0

    def __init__(self, function_id, speed, position, materials=0, products=0):
        super().__init__()
        # 需要有当前这个工作单元每个功能的备件，每个功能生产效率
        self.cell_id = WorkCell.next_id
        WorkCell.next_id += 1
        self.function = function_id
        self.speed = speed
        self.materials = materials
        self.products = products
        self.position = np.array(position)
        self.health = 100
        self.working = False
        self.idle_time = 0
        self.state = StateCode.workcell_ready

    def set_work(self, function):
        self.working = function
        self.state = StateCode.workcell_working

    def transport(self, action, num):
        # 转移生产产品
        if action == 2:
            self.products = 0

        # 或者接收原材料
        elif action == 3:
            self.materials += num

    def work(self, action):
        # 工作/继续工作
        if action == 1:
            if self.working:
                pass
            else:
                self.set_work(self.function)
                self.working = self.function
            self.idle_time = 0
        # 停止工作
        elif action == 0:
            # self.working = None
            self.working = None
            self.state = StateCode.workcell_ready
        # 检查当前状态
        self.state_check()
        if self.state == StateCode.workcell_working:
            # 工作中
            # 2是生产库存数量，0是生产速度,1是原料数量
            self.products += self.speed
            self.materials -= self.speed
            # self.health -= 0.1
        if self.state == StateCode.workcell_ready:
            self.idle_time += 1

    def state_check(self):
        # 低健康度
        # if self.health < 50:
        #     self.working = None
        #     self.state = StateCode.workcell_low_health
        # 缺少原料
        if self.materials < self.speed:
            self.state = StateCode.workcell_low_material
            self.working = None
        # 不缺货就变为ready状态
        elif (
            self.materials >= self.speed
            and self.state == StateCode.workcell_low_material
        ):
            self.state = StateCode.workcell_ready
        # 爆仓了
        if self.working is None:
            return

    def reset_state(self):
        self.state = StateCode.workcell_ready
        self.materials = self.speed
        self.products = 0
        self.working = None

    # 状态空间
    def get_state(self):
        if self.state == StateCode.workcell_working:
            return torch.tensor(
                [self.state.value, self.working, self.speed, self.products]
            )

        else:
            return torch.tensor(
                [self.state.value, self.function, self.speed, self.products]
            )

    # 动作空间
    def get_function(self):
        return self.function
