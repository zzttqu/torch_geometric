import math

from loguru import logger
import torch
from model.StateCode import *
from typing import ClassVar, Tuple
from BasicClass import BasicClass


class StorageCenter(BasicClass):

    def work(self, *args):
        pass

    def __init__(self, product_id: int, process_id: int, goal: int, max_func: int, process_num: int):
        """
            初始化产品中心
        Args:
            product_id:  产品id
            goal: 目标生产数量，用于规范化
            max_func: 最大功能数，用于规范化
        """
        super().__init__(process_id)
        self._product_id = product_id
        self.max_func = max_func if max_func > 1 else max_func
        self._product_count: int = 0

        self._is_last = process_id == process_num
        self._is_first = process_id == 0
        # 如果是process=0是原料仓库，库存就是产品数量*1.2，防止因为中间不够而卡住无法达到order数量
        if self._is_first:
            self._product_count = int(1.2 * goal)
        self.process_num = process_num
        self.goal = goal
        self.state = CellCode.workcell_working
        self.step_send_product = 0

    @property
    def is_last(self):
        return self._is_last

    @property
    def product_id(self):
        return self._product_id

    @property
    def product_count(self):
        return self._product_count

    def receive(self, num: int) -> None:
        """
        从生产中心接收产品
        :param num:接收产品数量
        """
        self._product_count += num
        #
        # logger.debug(
        #     f"{self.id}号货架，工序{self.process},储存{self.product_id}产品，接收产品{num}现有产品{self._product_count}")

    def send(self, ratio: float, speed: int) -> int:
        """
        从产品中心向下一级生产中心发送产品
        需要知道当前工作单元的工作能力，不发出超过一次speed的数量。
        :param
        ratio: 需要发送产品占比
        speed: 工作单元能力
        :return:
        """
        if self._product_count == 0:
            return 0
        product = math.floor(self._product_count * ratio)
        # 保证只要send就能send出一次工作需要的，防止囤积在cell中
        if product >= speed:
            product = speed
        else:
            product = 0
        self.step_send_product += product
        # if self.process == 0:
        #     logger.debug(
        #         f"{self.id}号货架，工序{self.process}，发送产品类别{self.product_id},{product}，{speed}发送比例:{ratio:.2f}现有产品{self._product_count - self.step_send_product}")
        return product

    # 等全部发送完了才能减掉
    def step(self):
        self._product_count = self._product_count - self.step_send_product
        self.step_send_product = 0

    def reset(self):
        self._product_count = 0
        if self._is_first:
            self._product_count = int(1.2 * self.goal)

    def status(self) -> torch.Tensor:
        """
        获取当前产品中心
        :return: 返回产品状态【规范化产品id，规范化生产数量】
        """
        # 改用生产进度作为表征
        # id归一化
        produce_progress = self._product_count / self.goal
        product_id_norm = self.product_id / (self.max_func - 1)
        process_norm = self.process / self.process_num
        return torch.tensor([product_id_norm, process_norm, produce_progress], dtype=torch.float32)

    def read_state(self) -> tuple[int, int]:
        """
        读取产品状态，不用规范化
        :return:【产品id，产品数量】
        """
        return self._product_count, self.product_id

    @property
    def is_first(self):
        return self._is_first
