# -*- coding: utf-8 -*-
# @Time     : 2019/4/10 14:32
# @Author   : Run 
# @File     : utils.py
# @Software : PyCharm


import time
import logging
import random
from collections import defaultdict
from string import ascii_lowercase
import numpy as np
import pandas as pd
from typing import Iterable


def gen_logger(logger_name: str = None) -> logging.Logger:
    """
    generate logger by Python standard library `logging`
    todo add other handlers
    Notes:
        1. recommend a third-party module `loguru`, more powerful and pleasant
    """
    # logger
    logger = logging.getLogger(str(random.random()))  # set random name to avoid influence between different loggers
    logger.setLevel(logging.DEBUG)  # set logger's level to the lowest, logging.NOTEST will cause strange situations.
    logger.name = logger_name

    # formatter
    if logger_name is None:
        formatter = logging.Formatter('[%(asctime)s] [%(levelname)s] %(message)s', datefmt='%Y/%m/%d %H:%M:%S')
    else:
        formatter = logging.Formatter('[%(asctime)s] [%(name)s~%(levelname)s] %(message)s', datefmt='%Y/%m/%d %H:%M:%S')

    # handlers
    # 1. print to screen
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(logging.WARNING)
    logger.addHandler(stream_handler)
    # # 2. print to file
    # file_handler = logging.FileHandler("output.log", encoding='UTF-8', mode='w')
    # file_handler.setFormatter(formatter)
    # file_handler.setLevel(logging.DEBUG)
    # logger.addHandler(file_handler)

    return logger


class Timer:
    """
    timing execution
    using `time.time()`

    Examples
    --------
    timer = Timer()
    <code_block1: to measure>
    timer.toc("block1")
    <code_block2: not to measure>
    timer.tic()
    <code_block3: to measure>
    timer.toc("block3")
    <code_block4: to measure>
    timer.toc("block4")
    timer.total()  # block1's time + block3's time + block4's time

    Notes:
        1. recommend a third-party module `PySnooper`
        2. https://blog.csdn.net/qq_27283619/article/details/89280974 distinguish time.time() with time.perf_counter()
    """

    def __init__(self, logger_func=None):
        self.index = 0
        self.records = []  # [(index, comments, cost_time), ...]
        self.groups = defaultdict(list)  # {group_name: [(index, comments, cost_time), ...], ...}
        self.logger_func = logger_func
        self.prev = time.time()

    def reset(self, drop_logger_func=False):
        self.index = 0
        self.records = []  # [(index, comments, cost_time), ...]
        self.groups = defaultdict(list)  # {group_name: [(index, comments, cost_time), ...], ...}
        if drop_logger_func:
            self.logger_func = None
        self.prev = time.time()

    def tic(self):
        self.prev = time.time()

    def toc(self, comments="", display=True, round_num=3):
        cost_time = round(time.time() - self.prev, round_num)
        self.records.append((self.index, comments, cost_time))
        if display:
            # print("{}. {} {}s".format(self.index, comments, cost_time))
            print("{} {}s".format(comments, cost_time))
            if self.logger_func:
                self.logger_func("{} {}s".format(comments, cost_time))
        self.index += 1
        self.prev = time.time()

    def add(self, group_name, comments="", round_num=3):
        cost_time = round(time.time() - self.prev, round_num)
        self.records.append((self.index, comments, cost_time))
        self.groups[group_name].append((self.index, comments, cost_time))
        self.index += 1
        self.prev = time.time()

    def summary(self, order=True, descending=True):
        """designed for groups"""
        res = [(k, len(v), sum(x[2] for x in v)) for k, v in self.groups.items()]
        if order:
            res = sorted(res, key=lambda x: x[2], reverse=descending)
        for group_name, num, cost_time in res:
            print("{}: {} splits, cost {}s".format(group_name, num, cost_time))

    def total(self):
        return sum(x[2] for x in self.records)

    def order(self, descending=True):
        """

        :param descending: bool
        :return:
        """
        return sorted(self.records, key=lambda x: x[2], reverse=descending)


def gen_df(row_num: int, col_num: int, lb: int = 0, ub: int = 100):
    """
    generate pd.DataFrame consist of int data and lowercase headers,
    mainly used to generate ordinary dataframes to test Cube in transform.py

    Parameters
    ----------
    row_num: int

    col_num: int

    lb: int, default = 0

    ub: int, default = 100

    Returns
    -------
    df: pd.DataFrame

    """
    shape = (row_num, col_num)
    data = np.random.randint(lb, ub, shape)
    cols = list(ascii_lowercase[:col_num])
    df = pd.DataFrame(data, columns=cols)
    print("shape:", df.shape)
    print(df.head(2))
    return df


def gen_ts_df():
    """
    mainly used to generate time-series dataframes to test functions in time_series.py
    e.g.
        层级：shop -> brand -> series -> sku 颗粒度由粗到细
        sku基本属性：price（价格，如果会随着日期改变，就将其移入销售数据内）
        销售数据：date, qty, amount(=qty*price)

    Returns
    -------

    """
    levels = ['shop', 'brand', 'series', 'sku']
    ratios = [2, 2, 3, 4]  # 2 shops, each has 2 brands,...
    start_date = '20190101'
    end_date = '20200930'
    date_range = pd.date_range(start_date, end_date)

    # basic info
    last_level = levels[0]
    last_level_objs = [f'{last_level}_{i}' for i in range(1, ratios[0] + 1)]
    basic_info_df = pd.DataFrame({levels[0]: last_level_objs})
    n = ratios[0]
    for level, ratio in zip(levels[1:], ratios[1:]):
        n *= ratio
        this_level_objs = [f'{level}_{i}' for i in range(1, n + 1)]
        map_df = pd.DataFrame({
            last_level: np.repeat(last_level_objs, ratio),
            level: this_level_objs
        })
        basic_info_df = pd.merge(basic_info_df, map_df)
        last_level = level
        last_level_objs = this_level_objs
    basic_info_df['price'] = np.random.randint(1, 100, n)

    sku_df_list = []
    # 对于每个sku，约30%的天数销量设置为0，其他时候在1-1000内随机生成
    for sku in last_level_objs:
        sku_df = pd.DataFrame({
            last_level: sku,
            'date': date_range,
            'qty': [0 if np.random.rand() < 0.3 else np.random.randint(1, 1000) for _ in range(len(date_range))]
        })
        # sku_df['amount'] = sku_df['qty'] * sku_df['price']
        sku_df_list.append(sku_df)
    sku_qty_df = pd.concat(sku_df_list, ignore_index=True)

    res_df = pd.merge(basic_info_df, sku_qty_df)
    res_df['amount'] = res_df['qty'] * res_df['price']

    return res_df


def flatten_iterable(its: Iterable, deep: bool = False) -> list:
    """
    flatten instance of Iterable to list
    Notes:
        1. except of str, won't flatten 'abc' to 'a', 'b', 'c'
    demo: [[[1], [2], [3]], 4]
        if deep is True: flatten to [1, 2, 3, 4]
        if deep is False, flatten to [[1], [2], [3], 4].
    """
    res = []
    for it in its:
        if isinstance(it, str):
            res.append(it)
        elif isinstance(it, Iterable):
            if deep:
                res += flatten_iterable(it, True)
            else:
                res.extend(it)
        else:
            res.append(it)
    return res


def compare_df(df1, df2, delta=1e-6) -> bool:
    """
    Check if two DataFrames are the same.

    Parameters
    ----------
    df1: pd.DataFrame

    df2: pd.DataFrame

    delta: float, default = 1e-6
        Precision of float.

    Returns
    -------
    flag: bool
        whether equals or not

    Notes
    -----
        1. pd.DataFrame自带equals和eq方法，也能用于比较两个DataFrame，但是对于不相等的情况缺乏解释，也没有考虑浮点数精度的影响，例如：
            >>> df1 = pd.DataFrame({'a': [0.3 - 0.1]})
            >>> df2 = pd.DataFrame({'a': [0.1 + 0.1]})
            >>> df1.equals(df2)
            False
            >>> df1.eq(df2)
                   a
            0  False
            事实上，我们是期望这两种方式计算的结果是被判定为相等的。
        2. 两列的dtypes相等，不代表具体位置的type也相等，例如：
            >>> df1 = pd.DataFrame({'a': [float('nan')]})
            >>> df1['a'] = df1['a'].astype(object)
            >>> df1.dtypes
            a    object
            dtype: object
            >>> type(df1['a'][0])
            float
            >>> df2 = pd.DataFrame({'a': [float('nan')]})
            >>> df2.dtypes
            a    float64
            dtype: object
            >>> type(df2['a'][0])
            numpy.float64
            >>> df1.equals(df2)
            False
            >>> compare_df(df1, df2)
            different dtypes
            df1: a    object
            dtype: object
            df2: a    float64
            dtype: object
            False
    """
    if df1.shape != df2.shape:
        print("different shape")
        print("df1: {}".format(df1.shape))
        print("df2: {}".format(df2.shape))
        return False
    if df1.dtypes.tolist() != df2.dtypes.tolist():
        print("different dtypes")
        print("df1: {}".format(df1.dtypes))
        print("df2: {}".format(df2.dtypes))
        return False
    if df1.columns.names != df2.columns.names:
        print("different header names")
        print("df1: {}".format(df1.columns.names))
        print("df2: {}".format(df2.columns.names))
        return False
    if df1.columns.tolist() != df2.columns.tolist():
        print("different header")
        print("df1: {}".format(df1.columns))
        print("df2: {}".format(df2.columns))
        return False
    if df1.index.names != df2.index.names:
        print("different index names")
        print("df1: {}".format(df1.index.names))
        print("df2: {}".format(df2.index.names))
        return False
    if df1.index.tolist() != df2.index.tolist():
        print("different index")
        return False
    #
    m, n = df1.shape
    for i in range(m):
        for j in range(n):
            x1, x2 = df1.iloc[i, j], df2.iloc[i, j]
            type1, type2 = type(x1), type(x2)
            if type1 != type2:
                print("different type")
                print("type(df1[{}, {}]): {}".format(i, j, type1))
                print("type(df2[{}, {}]): {}".format(i, j, type2))
                return False
            if isinstance(x1, float) or isinstance(x1, np.floating):
                if abs(x1 - x2) > delta:
                    print("different float value")
                    print("df1[{}, {}]: {}".format(i, j, x1))
                    print("df2[{}, {}]: {}".format(i, j, x2))
                    return False
            else:
                if str(x1) != str(x2):
                    print("different value")
                    print("df1[{}, {}]: {}".format(i, j, x1))
                    print("df2[{}, {}]: {}".format(i, j, x2))
                    return False
    print("df1 and df2 are the same.")
    return True


def merge_by(df1: pd.DataFrame, df2: pd.DataFrame, condition):
    """
    similar to conditional join in sql
    :param df1:
    :param df2:
    :param condition:
    :return:
    """
    pass  # todo
