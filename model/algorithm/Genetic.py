import random
from typing import Tuple

import torch
from torch import Tensor
from loguru import logger
from datetime import datetime

import cProfile

"""
orders = [[10, 20, 4], [60, 5, 5], [20, 10, 15]]
process_list = [0, 1, 2, 3, 4]
# 既然不用这个单元种类，那他的占用时间就按0算咯
process_time = [[5, 10, 15, 20, 12], [8, 12, 18, 0, 12], [3, 6, 0, 10, 8]]
RMT_units = [16, 10, 10, 15, 10]
total_time = torch.zeros((len(orders), len(process_list), len(orders[0])), dtype=torch.int32)
# 那我还不如干脆按工序分别进行优化
for order_id in range(len(orders)):
    for product_id in range(len(orders[0])):
        # 按顺序计算每个产品的当前环节所需的总时间
        # 每行是一个工序，每列是一个产品
        for process_id, time in enumerate(total_time[order_id]):
            time[product_id] = orders[order_id][product_id] * process_time[product_id][process_id]
# time0 = total_time[0]
# 第0个订单每个产品每个工序的用时
logger.info(total_time)


def norm_list(array: Tensor, product_num, _funcs_per_process_num):
    array = array.reshape((_funcs_per_process_num, product_num))
    for i in range(len(array)):
        array[i] = array[i] / torch.sum(array[i])
    return array


def generate_individual(product_num, _funcs_per_process_num, syngen_num):
    init_syngen = []
    for i in range(syngen_num):
        tmp = torch.random.rand(_funcs_per_process_num * product_num)
        tmp = norm_list(tmp, _funcs_per_process_num, product_num)
        init_syngen.append(tmp)
    return init_syngen


def total_time_cal(_process_list, _pops, order_time):
    product_using_time = torch.zeros(len(orders[0]))
    for i in range(len(product_using_time)):
        for j in range(len(_process_list)):
            raw_process_unit = _pops[j][i] * RMT_units[j]
            process_unit = raw_process_unit if raw_process_unit > 1 else 1
            product_using_time[i] += order_time[j][i] / int(process_unit)
    # 计算适应度，使用时间越少适应度越大，分子可以修改
    fitness = 10 - product_using_time.sum() / 1e2
    # print(f'{product_using_time.sum():.2f}')

    return product_using_time.sum(), fitness


def select(population: list[Tensor], fitness_values: Tensor) -> Tuple[list[Tensor], list]:
    norm_fits = (fitness_values - fitness_values.min() + 0.01) / (fitness_values.max() - fitness_values.min())
    total_fitness = sum(norm_fits)
    probabilities = [_norm_fit / total_fitness for _norm_fit in norm_fits]
    # print(f'{probabilities}')
    selected_indices = random.choices(range(len(population)), probabilities, k=len(population))
    return [population[i] for i in selected_indices], selected_indices


def fitness(_process_list, pops, order_time):
    fit = torch.zeros(len(pops))
    logger.info(pops)
    for i, ratios in enumerate(pops):
        _total_time, _fit = total_time_cal(_process_list, ratios, order_time)
        # print(_total_time)
        fit[i] = _fit
    return fit


# 变异
def mutate(individual, mutation_rate=0.1):
    for i in range(len(individual)):
        if random.random() < mutation_rate:
            individual[i] = random.uniform(0, 1)
    return individual


def crossover(parent1: Tensor, parent2: Tensor):
    parent1 = parent1.flatten()
    parent2 = parent2.flatten()
    crossover_point = random.randint(0, len(parent1) - 1)
    # parent1_index = torch.arange(0, crossover_point)
    # parent2_index = torch.arange(crossover_point, len(parent1))
    child1 = torch.concatenate([parent1[:crossover_point], parent2[crossover_point:]])
    child2 = torch.concatenate([parent1[crossover_point:], parent2[:crossover_point]])
    return child1, child2


population_size = 10
pops = generate_individual(len(orders[0]), len(process_list), population_size)
for _ in range(50):
    fit = fitness(process_list, pops, total_time[0])
    pops, pop_index = select(pops, fit)
    new_pops = []
    # 遗传变异
    for _ in range(population_size // 2):
        # 从选择后的种群中随机选择两个个体
        parents = random.sample(pops, k=2)
        children = crossover(parents[0], parents[1])
        for child in children:
            mutate(child)
            new_pops.append(norm_list(child, len(orders[0]), len(process_list)))
    # print(len(new_pops))
    pops = new_pops

"""


# 每个产品的用时

# 每个工序的用时（感觉没啥用）
# for i in range(process_using_time.shape[0]):
#     for j in range(len(orders[0])):
#         # 计算各个工序所用时间
#         # 其实应该也计算各个产品用的时间更重要
#         process_using_time[i] += time0[i][j] / (resource_allocation_ratios[i][j] * RMT_units[i])
class GeneticAlgorithm:
    def __init__(self, population_num, generation, order, process_speed, rmt_units):
        # 既然不用这个单元种类，那他的占用时间就按0算咯
        #  [[5, 10, 15, 20, 12], [8, 12, 18, 0, 12], [3, 6, 0, 10, 8]]
        self.process_speed = process_speed
        # orders = [[10, 20, 4], [60, 5, 5], [20, 10, 15]]
        self.order = order
        self.RMT_units = rmt_units
        self.population_num = population_num
        self.generation = generation
        self.process_num = len(process_speed[0])
        self.process_list = [i for i in range(self.process_num)]
        self.product_num = len(order)
        self.population = torch.zeros((self.population_num, self.product_num, self.process_num))
        # logger.debug(self.population.shape)
        self.fitness = torch.zeros(self.population_num)
        self.time = torch.zeros(self.population_num)
        self.best_solution = []
        self.best_fitness = 0
        self.shortest_time = 0
        self.eta_total_time = torch.zeros((self.process_num, self.product_num), dtype=torch.float32)
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
        array = array.reshape((self.process_num, self.product_num))
        # for i in range(len(array)):
        #     array[i] = array[i] / torch.sum(array[i])
        # 使用这种方式减少了百分之30的时间
        normalized_array = array.reshape((self.process_num, self.product_num)) / torch.sum(array, dim=1).unsqueeze(1)
        # sum_per_row = torch.sum(array, dim=1, keepdim=True)
        # normalized_array = array / sum_per_row
        return normalized_array

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
        process_units = torch.maximum(raw_process_units, torch.tensor(0.1))
        _process_units = torch.round(torch.maximum(raw_process_units, torch.tensor(1)))

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

        """
        # 这里只能计算当前个体的优势
        for i, pop in enumerate(self.population):
            product_using_time = torch.zeros(self.product_num)
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
            self.population[:] = self.population[torch.where(self.fitness == self.fitness.max())[0][0]]
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
        for i in range(len(individual)):
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
        child1 = torch.concatenate([parent1[:crossover_point], parent2[crossover_point:]])
        child2 = torch.concatenate([parent1[crossover_point:], parent2[:crossover_point]])
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


def main():
    # 既然不用这个单元种类，那他的占用时间就按0算咯

    process_speed = torch.tensor([[5, 10, 15, 20, 12], [8, 12, 18, torch.nan, 12], [3, 6, torch.nan, 10, 8]])
    orders = [[100, 200, 400], [60, 50, 50], [200, 100, 105]]
    rmt_units = torch.tensor([16, 10, 10, 15, 10])
    orders_num = len(orders)
    # 100种群数量100次演变从1300ms压缩到400ms
    pop_num = 100
    generation = 50

    _init_time = datetime.now()
    ga = GeneticAlgorithm(pop_num, generation, orders[0], process_speed, rmt_units)
    short_time, solution = ga.evolve()
    unit_num = torch.round(solution * rmt_units.unsqueeze(1)).to(dtype=torch.int)
    _end_time = datetime.now()
    cost_second = (_end_time - _init_time).seconds
    cost_micro = (_end_time - _init_time).microseconds
    ms_time = cost_second * 1000 + cost_micro / 1000
    logger.info(f"最优解时间：{short_time:.2f}，最优解：{unit_num}")

    logger.info(f"遗传算法求解最优解耗时：{cost_second}秒:{cost_micro / 1000:.2f}毫秒")
    logger.info(f"种群数量{pop_num}遗传算法求解最优解耗时：{ms_time:.2f}毫秒")


if __name__ == '__main__':
    # plt.rcParams['figure.dpi'] = 200
    # plt.rcParams['figure.figsize'] = (16, 10)
    # # 设置字体以便正确显示中文
    # plt.rcParams['font.sans-serif'] = ['FangSong']
    # # 正确显示连字符
    # plt.rcParams['axes.unicode_minus'] = False
    main()
    # [[3, 3, 10],
    #                                               [2, 2, 6],
    #                                               [4, 5, 0],
    #                                               [3, 0, 12],
    #                                               [2, 3, 5, ]]
