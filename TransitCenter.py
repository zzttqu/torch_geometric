import torch
from envClass import StateCode


class TransitCenter:
    next_id = 0

    def __init__(self, product_id, goal, max_func):
        self.max_func = max_func
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
        # id归一化
        produce_prograss = self.product_num / self.goal
        product_id_norm = self.product_id / self.max_func
        return torch.tensor([product_id_norm, produce_prograss], dtype=torch.float32)

    def read_state(self):
        """ 可视化状态 """
        return [self.product_id, self.product_num]
