import numpy as np
import pandas as pd
from loguru import logger


def data_generator(process_num, product_num, data_len) -> tuple[list, list, list]:
    """
    生成数据
    Args:
        data_len: 生成数量
        process_num: 工序数量
        product_num: 产品种类

    Returns:

    """
    speed_list = [_ for _ in range(data_len)]
    order_list = [_ for _ in range(data_len)]
    rmt_units_num_list = [_ for _ in range(data_len)]
    for i in range(data_len):
        speed = np.random.randint(5, 21, size=(process_num, product_num))
        # 百分之10的概率为nan
        mask = np.less(np.random.rand(process_num, product_num), 0.1)
        speed[mask] = 0
        speed_list[i] = speed
        # 生成orders
        order = np.random.randint(100, 1000, size=product_num)
        order_list[i] = order
        # 生成可重构单元
        rmt_units_num = np.random.randint(10, 50, size=process_num)
        rmt_units_num_list[i] = rmt_units_num
    return speed_list, order_list, rmt_units_num_list


def save_to_csv(data: tuple):
    speed = data[0]
    order = data[1]
    rmt_units_num = data[2]

    speed = np.vstack((speed, np.zeros((1, speed.shape[1]), dtype=np.int32)))
    # logger.info(speed.dtype)

    df_speed = pd.DataFrame(speed, columns=[f'process_{i + 1}' for i in range(speed.shape[1])])
    # df[[f'orders_{i + 1}' for i in range(len(order))]] = pd.DataFrame(order)
    df_order = pd.DataFrame([order], columns=[f'product_{i + 1}' for i in range(len(order))])
    df_rmt_units_num = pd.DataFrame([rmt_units_num], columns=[f'rmt_unit_{i + 1}' for i in range(len(rmt_units_num))])
    df_order.to_csv('order.csv', index=False, mode='a', header=False)
    df_speed.to_csv('speed.csv', index=False, mode='a', header=False)
    df_rmt_units_num.to_csv('rmt_units_num.csv', index=False, mode='a', header=False)


if __name__ == '__main__':
    save_to_csv(data_generator(10, 10))
