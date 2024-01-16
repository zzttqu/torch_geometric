import random
from typing import Tuple

import torch
from torch import Tensor
from torch import tensor, zeros
from loguru import logger
from datetime import datetime

import cProfile


# 每个产品的用时

# 每个工序的用时（感觉没啥用）
# for i in range(process_using_time.shape[0]):
#     for j in range(len(orders[0])):
#         # 计算各个工序所用时间
#         # 其实应该也计算各个产品用的时间更重要
#         process_using_time[i] += time0[i][j] / (resource_allocation_ratios[i][j] * RMT_units[i])
class GeneticAlgorithmTORCH:
    def __init__(self, population_num: int, generation: int, order: list, process_speed: list, rmt_units: list):
        # 既然不用这个单元种类，那他的占用时间就按0算咯
        #  [[5, 10, 15, 20, 12], [8, 12, 18, 0, 12], [3, 6, 0, 10, 8]]
        self.process_speed = tensor(process_speed)
        # orders = [[10, 20, 4], [60, 5, 5], [20, 10, 15]]
        self.order = order
        self.RMT_units = tensor(rmt_units)
        self.population_num = population_num
        self.generation = generation
        self.process_num = len(process_speed[0])
        self.process_list = [i for i in range(self.process_num)]
        self.product_num = len(order)
        self.population = zeros((self.population_num, self.product_num, self.process_num))
        # logger.debug(self.population.shape)
        self.fitness = zeros(self.population_num)
        self.time = zeros(self.population_num)
        self.best_solution = []
        self.best_fitness = 0
        self.shortest_time = 0
        self.eta_total_time = zeros((self.process_num, self.product_num), dtype=torch.float32)
        self.init_order()

    def init_order(self):
        # 那我还不如干脆按产品分别进行优化，一个产品一列
        for _product_id in range(self.product_num):
            # 按顺序计算每个产品的当前环节所需的总时间
            # 每行是一个工序，每列是一个产品
            for _process_id, _time in enumerate(self.eta_total_time):
                assert isinstance(_time, Tensor)
                if self.process_speed[_product_id][_process_id] != 0:
                    _time[_product_id] = self.order[_product_id] / self.process_speed[_product_id][_process_id]
                else:
                    _time[_product_id] = 0
        # time0 = total_time[0]
        # 第0个订单每个产品每个工序的用时
        # logger.info(self.eta_total_time)

    # 输入的是单行的，先转为5行3列的
    def norm_list(self, array: Tensor):
        # array = array.reshape((self.process_num, self.product_num))
        # for i in range(len(array)):
        #     array[i] = array[i] / torch.sum(array[i])

        # 使用这种方式减少了百分之30的时间
        array = array.view((self.process_num, self.product_num))
        array /= torch.sum(array, dim=1).unsqueeze(1)
        # sum_per_row = torch.sum(array, dim=1, keepdim=True)
        # normalized_array = array / sum_per_row
        return array

    def generate_init_population(self):
        init_syngen = []
        for i in range(self.population_num):
            tmp = torch.rand(self.process_num * self.product_num)
            tmp = self.norm_list(tmp)
            init_syngen.append(tmp)
        self.population = torch.stack(init_syngen, dim=0)

    def fitness_time_cal(self):
        # 重复计算的中间结果
        # 使用了数组广播，不用显式循环，再次加速！
        # 这个就是逐元素相乘，不是矩阵乘法
        self.RMT_units = self.RMT_units.view(1, -1, 1)
        raw_process_units = self.population * self.RMT_units
        # 确保每个工序至少有一个单位，保证最小值大于1，然后再四舍五入取整，不能取整，计算的就不准了
        process_units = torch.maximum(raw_process_units, tensor(0.1))
        _process_units = torch.round(torch.maximum(raw_process_units, tensor(1)))

        # 计算产品使用时间
        process_speed = self.process_speed.T
        # logger.info(self.eta_total_time)
        # logger.info(process_units * process_speed)
        # 需要忽略nan
        # total_time少一个维度需要加一个维度（就是种群数量维度）
        raw_fitness = torch.nansum(self.eta_total_time.unsqueeze(0) / (process_units * process_speed),
                                   dim=(1, 2))

        # logger.info(raw_fitness)

        # 计算适应度
        self.fitness = 10 - raw_fitness / 1e2
        self.time = torch.nansum(self.eta_total_time.unsqueeze(0) / _process_units, dim=(1, 2))

    def select(self):
        select_method = 'best'
        # 选择最好的
        if select_method == 'best':
            self.population[:] = self.population[torch.where(torch.eq(self.fitness, self.fitness.max()))[0][0]]
        # 选择最好的几个
        elif select_method == 'bests':
            sorted_indices = torch.argsort(self.fitness)
            # logger.debug(f'{sorted_indices},{self.fitness},{torch.sort(self.fitness)}')
            pop2 = self.population[sorted_indices]
            # 取出最后三个元素，也就是最好的三个元素
            good_pop = pop2[-3:]
            pop_indices = torch.randint(0, 3, self.population_num)
            # 重复赋值
            for i in range(self.population_num):
                self.population[i] = good_pop[pop_indices[i]]
            # self.population = self.population
        elif select_method == 'bang':
            # 规范化适应度
            norm_fits = (self.fitness - self.fitness.min() + 0.01) / (self.fitness.max() - self.fitness.min())
            total_fitness = norm_fits.sum()
            probabilities = [_norm_fit / total_fitness for _norm_fit in norm_fits]
            # 俄罗斯轮盘赌选择
            selected_indices = random.choices(range(self.population_num), probabilities, k=self.population_num)
            self.population = [self.population[i] for i in selected_indices]

    # 变异
    # 输入的为单行的nparray
    def mutate(self, individual, mutation_rate=0.1):
        # random_numbers = torch.rand(len(individual))
        # individual = torch.where(torch.less(random_numbers, mutation_rate), torch.rand(individual.shape[0]), individual)
        p = len(individual)
        # individual = map(lambda x: x if random.random() > mutation_rate else random.uniform(0, 1), individual)
        # individual = [x if random.random() > mutation_rate else random.uniform(0, 1) for x in individual]
        # individual=tensor(individual)
        for i in range(p):
            if random.random() < mutation_rate:
                individual[i] = random.uniform(0, 1)
        return individual

    # 返回值为单行的nparray
    def crossover(self, parent1: Tensor, parent2: Tensor) -> Tuple[Tensor, Tensor]:
        parent1 = parent1.flatten()
        parent2 = parent2.flatten()
        crossover_point = random.randint(0, len(parent1) - 1)
        # parent1_index = torch.arange(0, crossover_point)
        # parent2_index = torch.arange(crossover_point, len(parent1))
        child1 = torch.cat([parent1[:crossover_point], parent2[crossover_point:]])
        child2 = torch.cat([parent1[crossover_point:], parent2[:crossover_point]])
        return child1, child2

    def evolve(self):
        self.generate_init_population()
        for _ in range(self.generation):
            self.fitness_time_cal()
            self.select()
            self.best_solution = self.population[torch.where(self.fitness == self.fitness.max())[0][0]]
            self.best_fitness = self.fitness.max()
            self.shortest_time = self.time.min()
            new_pops = []
            # 遗传变异
            for _ in range(self.population_num // 2):
                # 从选择后的种群中随机选择两个个体,打乱index顺序（不重复），然后选择前两个
                parents_index = torch.randperm(self.population.shape[0])[:2]
                parents = self.population[parents_index]
                children = self.crossover(parents[0], parents[1])
                for child in children:
                    self.mutate(child)
                    new_pops.append(self.norm_list(child))
            # print(len(new_pops))
            self.population = torch.stack(new_pops, dim=0)
        # logger.info(
        #     # 最优解：{self.best_solution}，最优解适应度：{self.best_fitness: .2f}
        #     f"最优解时间：{self.shortest_time[-1]:.2f}")
        return self.shortest_time, self.best_solution


import numpy as np


class GeneticAlgorithmNUMPY:

    def __init__(self, population_num, generation, order: list, process_speed: list, rmt_units: list):

        # 既然不用这个单元种类，那他的占用时间就按0算咯
        #  [[5, 10, 15, 20, 12], [8, 12, 18, 0, 12], [3, 6, 0, 10, 8]]
        self.process_speed = np.array(process_speed)
        # orders = [[10, 20, 4], [60, 5, 5], [20, 10, 15]]
        self.order = order
        self.RMT_units = np.array(rmt_units)
        self.population_num = population_num
        self.generation = generation
        self.process_num = len(process_speed[0])
        self.process_list = [i for i in range(self.process_num)]
        self.product_num = len(order)
        self.population = np.zeros((self.population_num, self.product_num, self.process_num), dtype=np.float32)
        # logger.debug(self.population.shape)
        self.fitness = np.zeros(self.population_num)
        self.time = np.zeros(self.population_num)
        self.best_solution = []
        self.best_fitness = 0
        self.shortest_time = 0
        self.eta_total_time = np.zeros((self.process_num, self.product_num), dtype=np.float32)
        self.init_order()

    def init_order(self):
        # 那我还不如干脆按产品分别进行优化，一个产品一列
        for _product_id in range(self.product_num):
            # 按顺序计算每个产品的当前环节所需的总时间
            # 每行是一个工序，每列是一个产品
            for _process_id, _time in enumerate(self.eta_total_time):
                assert isinstance(_time, np.ndarray)
                if self.process_speed[_product_id][_process_id] != 0:
                    _time[_product_id] = self.order[_product_id] / self.process_speed[_product_id][_process_id]
                else:
                    _time[_product_id] = 0
        # time0 = total_time[0]
        # 第0个订单每个产品每个工序的用时
        # logger.info(self.eta_total_time)

    # 输入的是单行的，先转为5行3列的
    def norm_list(self, array: np.ndarray):
        array = array.reshape((self.process_num, self.product_num))
        # for i in range(len(array)):
        #     array[i] = array[i] / np.sum(array[i])
        # 使用这种方式减少了百分之30的时间
        sum_per_row = np.sum(array.reshape((self.process_num, self.product_num)), axis=1, keepdims=True)
        normalized_array = array / sum_per_row
        return normalized_array

    def generate_init_population(self):
        init_syngen = []
        for i in range(self.population_num):
            tmp = np.random.rand(self.process_num * self.product_num)
            tmp = self.norm_list(tmp)
            init_syngen.append(tmp)
        self.population = np.array(init_syngen)

    def fitness_time_cal(self):
        # 重复计算的中间结果
        # 使用了数组广播，不用显式循环，再次加速！
        # 这个就是逐元素相乘，不是矩阵乘法
        raw_process_units = self.population * self.RMT_units[np.newaxis, :, np.newaxis]
        # 确保每个工序至少有一个单位，保证最小值大于1，然后再四舍五入取整，不能取整，计算的就不准了
        process_units = np.maximum(raw_process_units, 0.1)
        _process_units = np.round(np.maximum(raw_process_units, 1))

        # 计算产品使用时间
        process_speed = self.process_speed.T
        # logger.info(self.eta_total_time)
        # logger.info(process_units * process_speed)
        # 需要忽略nan
        raw_fitness = np.nansum(self.eta_total_time[np.newaxis, :, :] / (process_units * process_speed),
                                axis=(1, 2))

        # logger.info(raw_fitness)

        # 计算适应度
        self.fitness = 10 - raw_fitness / 1e2
        self.time = np.nansum(self.eta_total_time[np.newaxis, :, :] / _process_units, axis=(1, 2))

        """
        # 这里只能计算当前个体的优势
        for i, pop in enumerate(self.population):
            product_using_time = np.zeros(self.product_num)
            for product in range(self.product_num):
                for process in range(self._funcs_per_process_num):
                    raw_process_unit = pop[process][product] * self.RMT_units[process]
                    process_unit = raw_process_unit if raw_process_unit > 1 else 1
                    product_using_time[product] += self.eta_total_time[process][product] / int(process_unit)
            # 计算适应度，使用时间越少适应度越大，分子可以修改
            self.fitness[i] = 10 - product_using_time.sum() / 1e2
            self.time[i] = product_using_time.sum()
        logger.info(self.time)"""

    def select(self):
        select_method = 'best'
        # 选择最好的
        if select_method == 'best':
            self.population[:] = self.population[np.where(self.fitness == self.fitness.max())[0][0]]
        # 选择最好的几个
        elif select_method == 'bests':
            sorted_indices = np.argsort(self.fitness)
            # logger.debug(f'{sorted_indices},{self.fitness},{np.sort(self.fitness)}')
            pop2 = self.population[sorted_indices]
            # 取出最后三个元素，也就是最好的三个元素
            good_pop = pop2[-3:]
            pop_indices = np.random.choice(np.arange(3, dtype=np.int32), size=self.population_num, replace=True)
            # 重复赋值
            for i in range(self.population_num):
                self.population[i] = good_pop[pop_indices[i]]
            # self.population = self.population
        elif select_method == 'bang':
            # 规范化适应度
            norm_fits = (self.fitness - self.fitness.min() + 0.01) / (self.fitness.max() - self.fitness.min())
            total_fitness = norm_fits.sum()
            probabilities = [_norm_fit / total_fitness for _norm_fit in norm_fits]
            # 俄罗斯轮盘赌选择
            selected_indices = random.choices(range(self.population_num), probabilities, k=self.population_num)
            self.population = [self.population[i] for i in selected_indices]

    # 变异
    # 输入的为单行的nparray
    def mutate(self, individual, mutation_rate=0.1):
        p = len(individual)
        # random_numbers = np.random.rand(len(individual))
        # np.where(random_numbers < mutation_rate, np.random.rand(individual.shape[0]), individual)
        for i in range(p):
            if random.random() < mutation_rate:
                individual[i] = random.uniform(0, 1)
        return individual

    # 返回值为单行的nparray
    def crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        parent1 = parent1.flatten()
        parent2 = parent2.flatten()
        crossover_point = random.randint(0, len(parent1) - 1)
        # parent1_index = np.arange(0, crossover_point)
        # parent2_index = np.arange(crossover_point, len(parent1))
        child1 = np.concatenate([parent1[:crossover_point], parent2[crossover_point:]])
        child2 = np.concatenate([parent1[crossover_point:], parent2[:crossover_point]])
        return child1, child2

    def evolve(self):
        self.generate_init_population()
        for _ in range(self.generation):
            self.fitness_time_cal()
            self.select()
            self.best_solution = self.population[np.where(self.fitness == self.fitness.max())[0][0]]
            self.best_fitness = self.fitness.max()
            self.shortest_time = self.time.min()
            new_pops = []
            # 遗传变异
            for _ in range(self.population_num // 2):
                # 从选择后的种群中随机选择两个个体
                parents_index = np.random.choice(self.population.shape[0], size=2, replace=False)
                parents = self.population[parents_index]
                children = self.crossover(parents[0], parents[1])
                for child in children:
                    self.mutate(child)
                    new_pops.append(self.norm_list(child))
            # print(len(new_pops))
            self.population = np.array(new_pops)
        # logger.info(
        #     # 最优解：{self.best_solution}，最优解适应度：{self.best_fitness: .2f}
        #     f"最优解时间：{self.shortest_time[-1]:.2f}")
        return self.shortest_time, self.best_solution


def mainTorch():
    # 既然不用这个单元种类，那他的占用时间就按0算咯
    with torch.no_grad():
        process_speed = [[5, 10, 15, 20, 12], [8, 12, 18, float('nan'), 12], [3, 6, float('nan'), 10, 8]]
        orders = [[100, 200, 400], [60, 50, 50], [200, 100, 105]]
        rmt_units = [16, 10, 10, 15, 10]
        orders_num = len(orders)
        # 100种群数量100次演变从1300ms压缩到400ms
        pop_num = 100
        generation = 50

        _init_time = datetime.now()
        ga = GeneticAlgorithmTORCH(pop_num, generation, orders[0], process_speed, rmt_units)
        short_time, solution = ga.evolve()
        _end_time = datetime.now()
        unit_num = [[round(j.item() * rmt_units[i]) for j in nums] for i, nums in enumerate(solution)]
        # unit_num = torch.round(solution * rmt_units.unsqueeze(1)).to(dtype=torch.int)
        cost_second = (_end_time - _init_time).seconds
        cost_micro = (_end_time - _init_time).microseconds
        ms_time = cost_second * 1000 + cost_micro / 1000
        # logger.info(f"最优解时间：{short_time:.2f}，最优解：{unit_num}")
        #
        # logger.info(f"遗传算法求解最优解耗时：{cost_second}秒:{cost_micro / 1000:.2f}毫秒")
        # logger.info(f"种群数量{pop_num}遗传算法求解最优解耗时：{ms_time:.2f}毫秒")


def mainNumpy():
    process_speed = [[5, 10, 15, 20, 12], [8, 12, 18, float('nan'), 12], [3, 6, float('nan'), 10, 8]]
    orders = [[100, 200, 400], [60, 50, 50], [200, 100, 105]]
    rmt_units = [16, 10, 10, 15, 10]
    orders_num = len(orders)
    # 100种群数量100次演变从1300ms压缩到400ms
    pop_num = 100
    generation = 50
    _init_time = datetime.now()
    ga = GeneticAlgorithmNUMPY(pop_num, generation, orders[0], process_speed, rmt_units)
    short_time, solution = ga.evolve()
    unit_num = [[round(j.item() * rmt_units[i]) for j in nums] for i, nums in enumerate(solution)]
    _end_time = datetime.now()
    cost_second = (_end_time - _init_time).seconds
    cost_micro = (_end_time - _init_time).microseconds
    ms_time = cost_second * 1000 + cost_micro / 1000
    # logger.info(f"最优解时间：{short_time:.2f}，最优解：{unit_num}")
    #
    # logger.info(f"遗传算法求解最优解耗时：{cost_second}秒:{cost_micro / 1000:.2f}毫秒")
    # logger.info(f"种群数量{pop_num}遗传算法求解最优解耗时：{ms_time:.2f}毫秒")


if __name__ == '__main__':
    # plt.rcParams['figure.dpi'] = 200
    # plt.rcParams['figure.figsize'] = (16, 10)
    # # 设置字体以便正确显示中文
    # plt.rcParams['font.sans-serif'] = ['FangSong']
    # # 正确显示连字符
    # plt.rcParams['axes.unicode_minus'] = False
    from timeit import timeit

    exit_time1 = timeit(mainTorch, number=100)
    exit_time2 = timeit(mainNumpy, number=100)
    logger.info(f'Torch:{exit_time1},Numpy:{exit_time2}')
    # mainTorch()
    # cProfile.run('main()', sort='cumulative')
    # [[3, 3, 10],
    #                                               [2, 2, 6],
    #                                               [4, 5, 0],
    #                                               [3, 0, 12],
    #                                               [2, 3, 5, ]]
