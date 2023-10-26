import torch
from envClass import StateCode


class TransitCenter:
    next_id = 0

    def __init__(self, product_id, goal):
        self.goal = goal
        self.cell_id = TransitCenter.next_id
        TransitCenter.next_id += 1
        self.product_id = product_id
        self.state = StateCode.workcell_working
        self.product_num = 0

    def recive_product(self, num: int) -> int:
        self.product_num += num
        return self.product_num

    def send_product(self, num: int) -> int:
        """
        从物流中心中向外发送产品
        """
        self.product_num -= num
        return self.product_num

    def get_product_num(self):
        return self.product_num

    def get_state(self):
        # 改用生产进度作为表征
        produce_prograss = self.product_num / self.goal
        return torch.tensor([self.product_id, produce_prograss], dtype=torch.float32)
