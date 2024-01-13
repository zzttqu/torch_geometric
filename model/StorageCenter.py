import math

from loguru import logger
import torch
from model.StateCode import *
from typing import ClassVar
from BasicClass import BasicClass


class StorageCenter(BasicClass):

    def work(self, *args):
        pass

    def __init__(self, product_id: int, process_id: int, goal: int, max_func: int):
        """
            初始化产品中心
        Args:
            product_id:  产品id
            goal: 目标生产数量，用于规范化
            max_func: 最大功能数，用于规范化
        """
        super().__init__(process_id)
        self._product_id = product_id
        self.max_func = 2 if max_func <= 1 else max_func

        self._product_count: int = 0
        # 如果是process=0是原料仓库，库存就是产品数量
        if self._process_id == 0:
            self._product_count = goal
        self.goal = goal
        self.state = StateCode.workcell_working
        self.step_send_product = 0

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

        product = math.floor(self._product_count * ratio)
        if product > speed:
            product = speed
        self.step_send_product += product
        return product

        # logger.debug(
        #     f"{self.id}号货架，工序{self.process}，发送产品{product}，发送比例:{ratio}现有产品{self._product_count - self.step_send_product}")

    # 等全部发送完了才能减掉
    def step(self):
        self._product_count = self._product_count - self.step_send_product
        self.step_send_product = 0

    def reset(self):
        self._product_count = 0
        if self._process_id == 0:
            self._product_count = self.goal

    def status(self) -> torch.Tensor:
        """
        获取当前产品中心
        :return: 返回产品状态【规范化产品id，规范化生产数量】
        """
        # 改用生产进度作为表征
        # id归一化
        produce_progress = self._product_count / self.goal
        product_id_norm = self.product_id / (self.max_func - 1)
        return torch.tensor([product_id_norm, produce_progress], dtype=torch.float32)

    def read_state(self) -> int:
        """
        读取产品状态，不用规范化
        :return:【产品id，产品数量】
        """
        return self._product_count
