# 这个类是用来定义加工中心的，一个加工中心包括多个加工单元，但同一时间只能有一个加工单元工作
class WorkCenter:
    next_id = 0

    def __init__(self,function_list,function) -> None:
        self.id = WorkCenter.next_id
        WorkCenter.next_id += 1
    
    def build_edge(self):
        # 建立内部节点的联系
        pass

    def work(self):
        pass
