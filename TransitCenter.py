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

    def putin_product(self, num):
        self.product_num += num
        return self.product_num

    def moveout_product(self, num):
        self.product_num -= num
        return self.product_num

    def get_state(self):
        return torch.tensor((self.state.value, self.product_id, 10, self.product_num))
