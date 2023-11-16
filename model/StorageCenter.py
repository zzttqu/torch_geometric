import torch
from model.StateCode import *
from typing import ClassVar


class StorageCenter:
    next_id: ClassVar = 0

    def __init__(self, product_id: int, goal: int, max_func: int):
        """
            初始化产品中心
        Args:
            product_id:  产品id
            goal: 目标生产数量，用于规范化
            max_func: 最大功能数，用于规范化
        """
        self.max_func = max_func
        self.goal = goal
        # self._id = StorageCenter.next_id
        self._id = int(product_id)
        StorageCenter.next_id += 1
        self.product_id = int(product_id)
        self.state = StateCode.workcell_working
        self.product_num = 0

    def get_id(self) -> int:
        return self._id

    def receive_product(self, num: int) -> None:
        """
        从生产中心接收产品
        :param num:接收产品数量
        """
        self.product_num += num

    def send_product(self, num: int) -> None:
        """
        从产品中心向下一级生产中心发送产品
        :param num: 需要发送产品数量
        :return:
        """
        self.product_num -= num

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
        product_id_norm = self.product_id / self.max_func
        return torch.tensor([product_id_norm, produce_progress], dtype=torch.float32)

    def read_state(self) -> list[int]:
        """
        读取产品状态，不用规范化
        :return:【产品id，产品数量】
        """
        return [self.product_id, self.product_num]
