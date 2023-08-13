from enum import Enum
import numpy as np
import networkx as nx
from matplotlib import pyplot as plt

graph = nx.Graph()


class StateCode(Enum):
    workcell_ready = 0
    workcell_start = 1
    workcell_function_err = 2
    workcell_busy = 3
    # workcell_finish = 4
    workcell_low_health = 5
    AGVCell_ready = 0
    AGVCell_start = 1
    AGVCell_busy = 2
    AGVCell_low_battery = 3
    AGVCell_overload = 4


class WorkCell:
    def __init__(self, cell_id: int, function_ids, function_times, position, health=100):
        super().__init__()
        self.cell_id = cell_id
        self.function = {}
        for function_id, time in zip(function_ids, function_times):
            self.function[function_id] = time
        self.position = np.array(position)
        self.health = 100
        self.working = False
        self.processing_time = 0
        self.idle_time = 0
        self.state = StateCode.workcell_ready

    def set_work(self, function):
        # 该工作站正在运行中，无法设置任务
        if self.state != StateCode.workcell_ready:
            self.state = StateCode.workcell_busy
        # 该工作站没有这种方法
        if function not in self.function:
            self.state = StateCode.workcell_function_err
        self.processing_time = self.function[function]
        self.working = True
        self.state = StateCode.workcell_start

    def work(self, action):
        if action[0] == 1:
            self.set_work(action[1])
            self.idle_time = 0
        state = self.state_check()
        # 工作中
        if state != StateCode.workcell_busy:
            self.state = state
        if self.processing_time <= 0:
            self.processing_time = 0
            self.state = StateCode.workcell_ready
        if self.state == StateCode.workcell_ready:
            self.idle_time += 1
        else:
            self.health -= 0.1
            self.processing_time -= 1

    def state_check(self):
        if self.working:
            return StateCode.workcell_busy
        if self.health < 50:
            return StateCode.workcell_low_health
        else:
            return StateCode.workcell_ready

    def reset_state(self):
        self.state = StateCode.workcell_ready
        self.working = False
        self.processing_time = 0

    def get_state(self):
        return self.state, self.processing_time


class AGVCell:
    def __init__(self, speed, capacity):
        self.speed = speed
        self.capacity = capacity
        self.health = 100
        self.battery = 100
        self.working = False
        # 运行到指定位置所需时间
        self.running_time = 0
        self.location = np.array([1, 2])

    def set_work(self, goal, payload):
        # 应该允许临时超载
        if payload > self.capacity:
            return StateCode.AGVCell_overload
        distance = np.sum(np.abs(goal - self.location))
        self.running_time = distance / self.speed
        # 如果电量不支持抵达就报错
        if self.running_time > self.battery:
            return StateCode.AGVCell_low_battery
        self.working = True
        return StateCode.AGVCell_start

    def work(self):
        if self.running_time > self.battery:
            return StateCode.AGVCell_low_battery
        self.running_time -= 1
        self.battery -= 1
        return StateCode.AGVCell_busy

    def state_check(self):
        if self.running_time > self.battery:
            return StateCode.AGVCell_low_battery
        if self.working:
            return StateCode.AGVCell_busy


class EnvRun:
    def __init__(self, step, work_cell_num, agv_cell_num):
        self.step = step
        self.work_cell_num = work_cell_num
        self.agv_cell_num = agv_cell_num
        self.work_cell_list = [
            WorkCell(cell_id=0, function_ids=[1, 4], function_times=[10, 12], position=[0, 0]),
            WorkCell(cell_id=1, function_ids=[2, 3], function_times=[5, 12], position=[0, 1]),
            WorkCell(cell_id=2, function_ids=[3, 4], function_times=[6, 2], position=[0, 2]),
            WorkCell(cell_id=3, function_ids=[1, 2], function_times=[6, 2], position=[1, 0]),
            WorkCell(cell_id=4, function_ids=[3, 4], function_times=[6, 2], position=[2, 1]),
        ]

    def build_edge(self):
        graph = nx.DiGraph()
        for i in range(len(self.work_cell_list)):
            for j in range(len(self.work_cell_list)):
                if i != j:
                    cell1_ids = self.work_cell_list[i].function.keys()
                    cell2_ids = self.work_cell_list[j].function.keys()
                    if any(id1 + 1 in cell2_ids for id1 in cell1_ids):
                        graph.add_edge(self.work_cell_list[i].cell_id, self.work_cell_list[j].cell_id)
        return graph


if __name__ == '__main__':
    env = EnvRun(1, 5, 1)
    graph = env.build_edge()
    pos = {}
    for i in range(5):
        pos[env.work_cell_list[i].cell_id] = env.work_cell_list[i].position
    nx.draw_networkx(graph, pos)
    plt.show()
