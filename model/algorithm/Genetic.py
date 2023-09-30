import random
from typing import Tuple
from loguru import logger
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import cProfile

np.set_printoptions(suppress=True)
"""
orders = [[10, 20, 4], [60, 5, 5], [20, 10, 15]]
process_list = [0, 1, 2, 3, 4]
# 既然不用这个单元种类，那他的占用时间就按0算咯
process_time = [[5, 10, 15, 20, 12], [8, 12, 18, 0, 12], [3, 6, 0, 10, 8]]
RMT_units = [16, 10, 10, 15, 10]
total_time = np.zeros((len(orders), len(process_list), len(orders[0])), dtype=np.int32)
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


def norm_list(array: np.ndarray, product_num, process_num):
    array = array.reshape((process_num, product_num))
    for i in range(len(array)):
        array[i] = array[i] / np.sum(array[i])
    return array


def generate_individual(product_num, process_num, syngen_num):
    init_syngen = []
    for i in range(syngen_num):
        tmp = np.random.rand(process_num * product_num)
        tmp = norm_list(tmp, process_num, product_num)
        init_syngen.append(tmp)
    return init_syngen


def total_time_cal(_process_list, _pops, order_time):
    product_using_time = np.zeros(len(orders[0]))
    for i in range(len(product_using_time)):
        for j in range(len(_process_list)):
            raw_process_unit = _pops[j][i] * RMT_units[j]
            process_unit = raw_process_unit if raw_process_unit > 1 else 1
            product_using_time[i] += order_time[j][i] / int(process_unit)
    # 计算适应度，使用时间越少适应度越大，分子可以修改
    fitness = 10 - product_using_time.sum() / 1e2
    # print(f'{product_using_time.sum():.2f}')

    return product_using_time.sum(), fitness


def select(population: list[np.ndarray], fitness_values: np.ndarray) -> Tuple[list[np.ndarray], list]:
    norm_fits = (fitness_values - fitness_values.min() + 0.01) / (fitness_values.max() - fitness_values.min())
    total_fitness = sum(norm_fits)
    probabilities = [_norm_fit / total_fitness for _norm_fit in norm_fits]
    # print(f'{probabilities}')
    selected_indices = random.choices(range(len(population)), probabilities, k=len(population))
    return [population[i] for i in selected_indices], selected_indices


def fitness(_process_list, pops, order_time):
    fit = np.zeros(len(pops))
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


def crossover(parent1: np.ndarray, parent2: np.ndarray):
    parent1 = parent1.flatten()
    parent2 = parent2.flatten()
    crossover_point = random.randint(0, len(parent1) - 1)
    # parent1_index = np.arange(0, crossover_point)
    # parent2_index = np.arange(crossover_point, len(parent1))
    child1 = np.concatenate([parent1[:crossover_point], parent2[crossover_point:]])
    child2 = np.concatenate([parent1[crossover_point:], parent2[:crossover_point]])
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
    def __init__(self, population_num, generation, order, process_time):
        # 既然不用这个单元种类，那他的占用时间就按0算咯
        #  [[5, 10, 15, 20, 12], [8, 12, 18, 0, 12], [3, 6, 0, 10, 8]]
        self.process_time = process_time
        # orders = [[10, 20, 4], [60, 5, 5], [20, 10, 15]]
        self.order = order
        self.RMT_units = np.array([16, 10, 10, 15, 10])
        self.population_num = population_num
        self.generation = generation
        self.process_num = len(process_time[0])
        self.process_list = [i for i in range(self.process_num)]
        self.product_num = len(order)
        self.population = np.zeros((self.population_num, self.product_num, self.process_num))
        # logger.debug(self.population.shape)
        self.fitness = np.zeros(self.population_num)
        self.time = np.zeros(self.population_num)
        self.best_solution = []
        self.best_fitness = 0
        self.shortest_time = []
        self.eta_total_time = np.zeros((self.process_num, self.product_num), dtype=np.int32)
        self.init_order()

    def init_order(self):
        # 那我还不如干脆按产品分别进行优化，一个产品一列
        for _product_id in range(self.product_num):
            # 按顺序计算每个产品的当前环节所需的总时间
            # 每行是一个工序，每列是一个产品
            for _process_id, _time in enumerate(self.eta_total_time):
                assert isinstance(_time, np.ndarray)
                _time[_product_id] = self.order[_product_id] * self.process_time[_product_id][_process_id]
        # time0 = total_time[0]
        # 第0个订单每个产品每个工序的用时
        # logger.info(self.eta_total_time)

    # 输入的是单行的，先转为5行3列的
    def norm_list(self, array: np.ndarray):
        array = array.reshape((self.process_num, self.product_num))
        # for i in range(len(array)):
        #     array[i] = array[i] / np.sum(array[i])
        # 使用这种方式减少了百分之30的时间
        sum_per_row = np.sum(array, axis=1, keepdims=True)
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
        # 确保每个工序至少有一个单位，保证最小值大于1，然后再四舍五入取整
        process_units = np.round(np.maximum(raw_process_units, 1))

        # 计算产品使用时间
        # logger.info(self.eta_total_time[np.newaxis, :, :] / process_units)
        product_using_time = np.sum(self.eta_total_time[np.newaxis, :, :] / process_units, axis=(1, 2))
        # logger.info(product_using_time)

        # 计算适应度
        self.fitness = 10 - product_using_time / 1e2
        self.time = product_using_time

        """
        # 这里只能计算当前个体的优势
        for i, pop in enumerate(self.population):
            product_using_time = np.zeros(self.product_num)
            for product in range(self.product_num):
                for process in range(self.process_num):
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
        for i in range(len(individual)):
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
            self.shortest_time.append(self.time.min())
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
        return self.shortest_time


def main():
    # 既然不用这个单元种类，那他的占用时间就按0算咯
    #  [[5, 10, 15, 20, 12], [8, 12, 18, 0, 12], [3, 6, 0, 10, 8]]
    init_time = datetime.now()
    process_time = [[5, 10, 15, 20, 12], [8, 12, 18, 0, 12], [3, 6, 0, 10, 8]]
    orders = [[10, 20, 4], [60, 5, 5], [20, 10, 15]]
    orders_num = len(orders)
    # 100种群数量100次演变从1300ms压缩到400ms
    pop_num = 10
    test_round = 5
    generation = 50
    si_time = np.zeros((orders_num, test_round))
    width = 0.4
    for i in range(orders_num):
        ga = GeneticAlgorithm(pop_num, generation, orders[i], process_time)
        pp = ga.eta_total_time.T
        plt.subplot(3, orders_num, i + 1)
        x = np.arange(5)
        plt.bar(x, pp[0], width=width, label='产品1')
        plt.bar(x, pp[1], width=width, label='产品2', bottom=pp[0])
        plt.bar(x, pp[2], width=width, label='产品3', bottom=pp[0] + pp[1])
        max_v = pp.T.sum(axis=1).max()
        plt.ylim(0, max_v + max_v / 10)
        # 在每个方块上边写上数值
        for j, product in enumerate(pp.T):
            if product[0] != 0:
                plt.text(j, product[0] / 2, str(product[0]), ha='center', va='center', color='black', fontweight='bold')
            if product[1] != 0:
                plt.text(j, product[0] + product[1] / 2, str(product[1]), ha='center', va='center', color='black',
                         fontweight='bold')
            if product[2] != 0:
                plt.text(j, product[0] + product[1] + product[2] / 2, str(product[2]), ha='center', va='center',
                         color='black', fontweight='bold')
            plt.text(j, product[0] + product[1] + product[2] + max_v / 20, str(sum(product)), ha='center', va='center',
                     color='black', fontweight='bold')
        plt.xticks(x, [f'工序{i}' for i in range(1, 6)])
        plt.legend()
        plt.title(f'订单{i + 1}工时统计')

    for j in range(test_round):
        pop_num = (j + 1) * 10
        init_time = datetime.now()
        for i in range(orders_num):
            _init_time = datetime.now()
            ga = GeneticAlgorithm(pop_num, generation, orders[i], process_time)
            short_time = ga.evolve()

            plt.subplot(3, orders_num, orders_num + i + 1)
            plt.title(f"订单{i + 1}")
            plt.xlabel("迭代次数")
            plt.ylabel("所用总时间步")
            plt.plot(short_time, label=f"种群为{pop_num}")
            plt.legend()

            _end_time = datetime.now()
            _cost_micro = (_end_time - _init_time).microseconds
            _cost_second = (_end_time - _init_time).seconds
            _ms_time = _cost_second * 1000 + _cost_micro / 1000
            si_time[i, j] = _ms_time
        end_time = datetime.now()
        cost_micro = (end_time - init_time).microseconds
        cost_second = (end_time - init_time).seconds
        ms_time = cost_second * 1000 + cost_micro / 1000
        # logger.info(f"遗传算法求解最优解耗时：{cost_second}秒:{cost_micro / 1000:.2f}毫秒")
        # logger.info(f"种群数量{pop_num}遗传算法求解最优解耗时：{ms_time:.2f}毫秒")

    for q in range(orders_num):
        plt.subplot(3, orders_num, orders_num * 2 + q + 1)
        plt.title(f"订单{q + 1}")
        plt.xlabel("种群数量/个")
        plt.ylabel("所用时间/ms")
        x = [f'{i}' for i in range(10, 51, 10)]
        # 同一个i是同一个generation
        plt.bar(x, si_time[q], width=width)

    # for i in range(orders_num):
    #     ga = GeneticAlgorithm(50, 50, orders[i], process_time)
    #     short_time = ga.evolve()
    #     plt.subplot(2, orders_num, i + 1 + orders_num)
    #     plt.title(f"种群为50订单{i + 1}")
    #     plt.xlabel("迭代次数")
    #     plt.ylabel("所用时间")
    #     plt.plot(short_time)
    plt.tight_layout()
    plt.show()


# end_time = datetime.now()
# cost_micro = (end_time - init_time).microseconds
# cost_second = (end_time - init_time).seconds
# # logger.info(f"遗传算法求解最优解耗时：{cost_second}秒:{cost_micro / 1000:.2f}毫秒")
# logger.info(f"遗传算法求解最优解耗时：{cost_second * 1000 + cost_micro / 1000:.2f}毫秒")


if __name__ == '__main__':
    plt.rcParams['figure.dpi'] = 200
    plt.rcParams['figure.figsize'] = (16, 10)
    # 设置字体以便正确显示中文
    plt.rcParams['font.sans-serif'] = ['FangSong']
    # 正确显示连字符
    plt.rcParams['axes.unicode_minus'] = False
    main()
    # cProfile.run('main()', sort='cumulative')
