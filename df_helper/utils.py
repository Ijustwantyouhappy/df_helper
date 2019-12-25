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


def gen_df(row_num: int, col_num: int, lb: int = 0, ub: int = 100):
    """
    generate pd.DataFrame consist of int data and lowercase headers
    :param row_num:
    :param col_num:
    :param lb: lower bound
    :param ub: upper bound
    :return:
    """
    shape = (row_num, col_num)
    data = np.random.randint(lb, ub, shape)
    cols = list(ascii_lowercase[:col_num])
    df = pd.DataFrame(data, columns=cols)
    print("shape:", df.shape)
    print(df.head(2))
    return df


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
