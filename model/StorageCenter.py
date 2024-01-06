import math

import torch
from model.StateCode import *
from typing import ClassVar


class StorageCenter:
    _next_id = 0

    def __init__(self, product_id: int, process: int, goal: int, max_func: int):
        """
            初始化产品中心
        Args:
            product_id:  产品id
            goal: 目标生产数量，用于规范化
            max_func: 最大功能数，用于规范化
        """
        self.__id = StorageCenter.get_next_id()
        self._product_id = product_id
        self._process = process
        self.max_func = 2 if max_func <= 1 else max_func
        self.goal = goal
        self.state = StateCode.workcell_working
        self.product_num = 0

    @property
    def id(self):
        return self.__id

    @property
    def product_id(self):
        return self._product_id

    @property
    def process(self):
        return self._process

    @classmethod
    def get_next_id(cls):
        current_id = cls._next_id
        cls._next_id += 1
        return current_id

    @classmethod
    def reset_id(cls):
        cls._next_id = 0

    def receive_product(self, num: int) -> None:
        """
        从生产中心接收产品
        :param num:接收产品数量
        """
        self.product_num += num

    def send_product(self, ratio: int) -> int:
        """
        从产品中心向下一级生产中心发送产品
        :param ratio: 需要发送产品占比
        :return:
        """
        product = math.floor(self.product_num * ratio)
        self.product_num -= product
        return product

    def get_product_num(self) -> int:
        """
        获取当前产品中心的产品数量
        :return: 返回产品数量
        """
        return self.product_num

    def get_state(self) -> torch.Tensor:
        """
        获取当前产品中心
        :return: 返回产品状态【规范化产品id，规范化生产数量】
        """
        # 改用生产进度作为表征
        # id归一化
        produce_progress = self.product_num / self.goal
        product_id_norm = self.product_id / (self.max_func - 1)
        return torch.tensor([product_id_norm, produce_progress], dtype=torch.float32)

    def read_state(self) -> list[int]:
        """
        读取产品状态，不用规范化
        :return:【产品id，产品数量】
        """
        return [self.product_id, self.product_num]
