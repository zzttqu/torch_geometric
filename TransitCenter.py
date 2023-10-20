import torch
from envClass import StateCode


class TransitCenter:
    next_id = 0

    def __init__(self, product_id):
        self.cell_id = TransitCenter.next_id
        TransitCenter.next_id += 1
        self.product_id = product_id
        self.state = StateCode.workcell_working
        self.product_num = 0

    def recive_product(self, num: int) -> int:
        self.product_num += num
        return self.product_num

    def send_product(self, num: int) -> int:
        self.product_num -= num
        return self.product_num

    def get_state(self):
        return torch.tensor((self.product_id, self.state.value, self.product_num))
