# -*- coding: utf-8 -*-
# @Time     : 2019/12/26 0:34
# @Author   : Run
# @File     : time_series.py
# @Software : PyCharm

"""
mainly used to extract features from pd.DataFrame which contains a column recording date information
todo 目前着重于功能的实现，没有太多的考虑效率。

Notes:
"""

# import os
import sys
from itertools import product
import numpy as np
import pandas as pd
# from df_helper.utils import gen_logger

# TIME_SERIES_LOGGER = gen_logger(os.path.basename(__file__))


def add_date_attrs(df, attrs, names=None, date_col='date', inplace=False):
    """
    添加和时间有关的绝对属性列。
    仅和date_col有关，如日期2020-09-29的year是2020，month是9，day是28，quarter是3，year_month是202009

    Examples
    --------
    >>> from df_helper.utils import gen_ts_df
    >>> df = gen_ts_df()
    >>> df = add_date_attrs(df, ['year', 'quarter', 'month', 'month_name', 'year_month'])

    Parameters
    ----------
    df: pd.DataFrame

    attrs: list of str
        - Attributes in `pdir(pd.Series.dt)`
        - year_month: 新添一列表示年月，如202006
        - season/season_name: todo 待完善

    names: list of str, default = None
        column name for attr in attrs, if None, use attrs as names directly

    date_col: str, default = 'date'

    inplace: bool, default = True

    Returns
    -------

    Notes
    -----
    1. pd.Series.dt的week属性和weekofyear属性的效果一致，都是指该日期所在周是一年的第几周。
        注意同一周的日期该属性总是一致的，所以当一周跨年时，该周位于哪一年内的天数多，则归于哪一年。
        例如，2016-12-26至2017-01-01该属性均为52，而2018-12-31至2019-01-06该属性均为1。
    2. pd.Series.dt的weekday属性和dayofweek属性的效果一致，周一到周日分别为0-6
    """
    dt_obj = pd.to_datetime(df[date_col]).dt  # datetime.date格式不支持取dt的操作，要先转成datetime.datetime
    df_cols = set(df.columns)
    extra_attrs = [
        'year_month',
        # 'season', 'season_name',  # todo
    ]

    if names is None:
        names = attrs
    else:
        assert len(attrs) == len(set(attrs)), "duplicates in attrs"
        assert len(names) == len(set(names)), "duplicates in names"
        assert len(attrs) == len(names), "attrs and names are unmatched"

    attrs_filtered = []
    existed_names = []
    for attr, name in zip(attrs, names):
        if not (hasattr(dt_obj, attr) or attr in extra_attrs):
            attrs_filtered.append(attr)
        if name in df_cols:
            existed_names.append(name)
    assert len(attrs_filtered) == 0, f"invalid attributes: {attrs_filtered}"
    assert len(existed_names) == 0, f"already existed names: {existed_names}"
    # flag = False
    # if attrs_filtered:
    #     print(f"[{sys._getframe().f_code.co_name}] invalid attributes: {attrs_filtered}")
    #     # TIME_SERIES_LOGGER.error(f"[{sys._getframe().f_code.co_name}] invalid attributes: {attrs_filtered}")
    #     flag = True
    # if existed_cols:
    #     print(f"[{sys._getframe().f_code.co_name}] columns `{existed_cols}` already exists, please check.")
    #     # TIME_SERIES_LOGGER.error(
    #     #     f"[{sys._getframe().f_code.co_name}] columns `{existed_cols}` already exists, please check.")
    #     flag = True
    # if flag:
    #     return

    if not inplace:
        df = df.copy()
    for attr, name in zip(attrs, names):
        if hasattr(dt_obj, attr):
            obj = getattr(dt_obj, attr)
            if callable(obj):
                obj = obj()
            df[name] = obj
        elif attr == 'year_month':
            df[name] = dt_obj.year * 100 + dt_obj.month

    return df


def add_date_index(df, level='day', col_name=None, base_date=None, date_col='date', inplace=True):
    """
    add integer index for date_col
    - 年份的index就是其本身；
    - 其他的时间颗粒度的index则是以base_date为0，后续日期按与base_date的差额计算。（通常要求base_date不能晚于df中date_col列内最早的日期）
        - quarter:
        - month: e.g. base_date='2018-01-01'，则2018-01-01 ~ 2018-01-31的month_idx均为0，2018年2月内各日期的month_index均为1，
            而2019-01-01的month_index为12，以此类推。
        - week:
        - day:

    Examples
    --------
    >>> from df_helper.utils import gen_ts_df
    >>> df = gen_ts_df()
    >>> df = add_date_index(df, 'day')

    Parameters
    ----------
    df: pd.DataFrame

    level: str, default = 'day'
        想要加的index的颗粒度，可以为：day/week/month/quarter/year

    col_name: str, default = None
        生成的index列的列名，如果不传，则默认为 f'{level}_index'

    base_date: str/datetime/date, default = None
        - None: 默认取date_col列内最早的日期；
        - 如果传入，要保证不能晚于df中date_col列内最早的日期

    date_col: str, default = 'date'
        日期列的列名

    inplace: bool, default = True

    Returns
    -------

    Warnings
    --------
    1. year的index总是为其自身；除此外，其他时间颗粒度的index是一个相对的概念，改变base_date会影响到生成的index的数值

    """
    valid_levels = ['year', 'quarter', 'month', 'week', 'day']
    assert level in valid_levels, "invalid level"

    if col_name is None:
        col_name = f'{level}_index'
    assert col_name not in df.columns, f"{level}_index already existed"
    # if col_name in df.columns:
    #     print(f"[{sys._getframe().f_code.co_name}] column `{col_name}` already exists, please check.")
    #     # TIME_SERIES_LOGGER.error(
    #     #     f"[{sys._getframe().f_code.co_name}] column `{col_name}` already exists, please check.")
    #     return

    date_ser = pd.to_datetime(df[date_col])
    dt_obj = date_ser.dt
    if base_date is None:
        start_date = date_ser.min()
    else:
        start_date = pd.to_datetime(base_date)

    if not inplace:
        df = df.copy()

    if level == 'day':
        df[col_name] = (date_ser - start_date).dt.days
    elif level == 'week':  # 比较特别，详见函数add_date_attrs中的Notes 1.
        end_date = date_ser.max()
        dates = pd.date_range(start_date, end_date)
        dates_len = len(dates)
        starts = [0] * (7 - start_date.dayofweek)
        starts_len = len(starts)
        if dates_len < starts_len:
            indexes = starts[:dates_len]
        else:
            n, m = divmod(dates_len - starts_len, 7)
            indexes = starts + list(np.repeat(range(1, n + 1), 7)) + [n + 1] * m
        mapping_d = dict(zip(dates, indexes))
        df[col_name] = date_ser.map(mapping_d)
    elif level == 'month':
        df[col_name] = (dt_obj.year - start_date.year) * 12 + (dt_obj.month - start_date.month)
    elif level == 'quarter':
        df[col_name] = (dt_obj.year - start_date.year) * 4 + (dt_obj.quarter - start_date.quarter)
    elif level == 'year':
        df[col_name] = dt_obj.year

    return df


def add_date_indexes(df, levels, names=None, base_date=None, date_col='date', inplace=True):
    """
    add multiple integer index for date_col
    详见函数add_date_index的函数文档

    Examples
    --------
    >>> from df_helper.utils import gen_ts_df
    >>> df = gen_ts_df()
    >>> df = add_date_indexes(df, ['day', 'week', 'month', 'quarter', 'year'])

    Parameters
    ----------
    df
    levels
    names
    base_date
    date_col
    inplace

    Returns
    -------

    """
    if names is None:
        names = [None] * len(levels)
    assert len(levels) == len(names), "levels and names are unmatched"
    for level, name in zip(levels, names):
        df = add_date_index(df, level, name, base_date, date_col, inplace)
    return df


def _gen_col_func_tuple_list(target_cols, agg_funcs):
    # called inside function `add_last_k_continuous_time_unit_statistics`
    if isinstance(target_cols, str):
        if isinstance(agg_funcs, str):
            col_func_tuple_list = [(target_cols, agg_funcs)]
        elif type(agg_funcs) in {set, list, tuple}:
            col_func_tuple_list = [(target_cols, func) for func in agg_funcs]
        else:
            print("please check type of parameters `target_cols` and `agg_funcs`")
            return
    elif type(target_cols) in {set, list, tuple}:
        if isinstance(agg_funcs, str):
            col_func_tuple_list = [(col, agg_funcs) for col in target_cols]
        elif type(agg_funcs) in {set, list, tuple}:
            col_func_tuple_list = list(product(target_cols, agg_funcs))
        else:
            print("please check type of parameters `target_cols` and `agg_funcs`")
            return
    elif isinstance(target_cols, dict):
        col_func_tuple_list = []
        for col, funcs in target_cols.items():
            if isinstance(funcs, str):
                col_func_tuple_list.append((col, funcs))
            elif type(funcs) in {set, list, tuple}:
                col_func_tuple_list += [(col, func) for func in funcs]
            else:
                print("please check type of parameters `target_cols` and `agg_funcs`")
                return
    else:
        print("please check type of parameters `target_cols` and `agg_funcs`")
        return
    return col_func_tuple_list


def add_last_k_continuous_time_unit_statistics(
        df, k, time_unit, target_cols, agg_funcs=None,
        date_index_col=None, id_col=None,
        ignore_start_units=True, shift_1_unit_backward=True):
    """
    按id_col分组成不同的时间序列，对target_cols的每个时间节点（time_unit层级）计算前k个连续时间单位内所有数据的统计量。
    例如：计算每个sku每个月的前3个月销量的均值。id_col传入sku列的列名，k传入3，time_unit传入'month'，
        target_cols传入销量列的列名，agg_func传入'mean'。如果只有一个sku则不需要传入id_col。

    Parameters
    ----------
    df : 要求：
        1. 必须含有time_unit层级的date_index（因为查找前k个连续时间单位是依据date_index进行的）
        2. target_cols中的列最好不要存在缺失值（极端情况如某个时间段内全为缺失值的话，无法计算统计量）
    k : int
    time_unit : day/week/month/quarter/year
    target_cols : 该参数和agg_funcs参数一起决定了为哪些列计算哪些统计量
        - str:
            * str: single statistic of single column
            * iterable: multi statistics of single column
        - list/set/tuple:
            * str: single statistic of multi column
            * iterable: multi statistic of multi column
        - dict: 当target_cols传入dict类型时，agg_funcs为None，即使有输入值也不生效
            * {col1: func11, col2: iterable(func21, ...), ...}
    agg_funcs : None/str/iterable. usual functions： sum/mean/median/min/max/var
    date_index_col: time_unit层级的date_index列的列名。如无传入，则默认为f"{time_unit}_index"
    id_col : 如果没有指定，则表明传入的DataFrame是一个时间序列；
             如果指定，则按该列分组后再为每个时间节点计算目标统计量
    ignore_start_units : 分组后，每组内前k-1个时间单位是否不计算统计量
        todo 对各个统计量做对应调整，如均值不变，sum按比例扩大等。
    shift_1_unit_backward : 返回的结果中对应关系是否向后错位了一个时间单位。
        如果为True，则第k+1个时间节点对应着1至k个月数据计算的统计量，
        如果为False，则第k个时间节点对应着1至k个月数据计算的统计量。

    Returns
    -------
    res_df : pd.DateFrame, columns: (id_col), date_index_col, f'last_{k}_continuous_{time_unit}_{col}_{func}'

    """
    if date_index_col is None:
        date_index_col = f"{time_unit}_index"
    if date_index_col not in df.columns:
        print(f"please input date_index_col of {time_unit} level")
        return

    col_func_tuple_list = _gen_col_func_tuple_list(target_cols, agg_funcs)
    if col_func_tuple_list is None:
        return

    if id_col is None:
        grouped = [(None, df)]
    else:
        grouped = df.groupby(id_col)

    res_df_list = []
    for sku, sku_df in grouped:
        start, end = sku_df[date_index_col].min(), sku_df[date_index_col].max()
        if ignore_start_units:
            start += k - 1
        data = []
        for m in range(start, end + 1):
            m_df = sku_df[(sku_df[date_index_col] >= m - k + 1) & (sku_df[date_index_col] <= m)]
            if m_df.empty:
                row = [None] * len(col_func_tuple_list)
            else:
                row = [getattr(m_df[col], func)() for col, func in col_func_tuple_list]
            data.append(row)
        prefix = f'last_{k}_continuous_{time_unit}_'
        cols = [prefix + f"{col}_{func}" for col, func in col_func_tuple_list]
        sub_df = pd.DataFrame(data, columns=cols)
        sub_df[date_index_col] = range(start, end + 1)
        if shift_1_unit_backward:
            sub_df[date_index_col] += 1
        if sku is not None:
            sub_df[id_col] = sku
        res_df_list.append(sub_df)

    res_df = pd.concat(res_df_list, ignore_index=True)
    return res_df


def add_last_k_discrete_time_unit_statistics(
        df, k, time_unit, target_cols, agg_funcs=None,
        date_index_col=None, id_col=None):
    """
    按id_col分组成不同的时间序列，对target_cols的每个时间节点（time_unit层级）计算前k个连续时间单位内各自数据的统计量。
    注意： 在每组内时间轴的两端各k个时间节点的统计量中会有缺失值出现，这是自然的。
    例如：计算每个sku每个月的上个月、上上个月、上上上个月各自销量的均值。id_col传入sku列的列名，k传入3，time_unit传入'month'，
        target_cols传入销量列的列名，agg_func传入'mean'。如果只有一个sku则不需要传入id_col。

    Parameters
    ----------
    df : 要求：
        1. 必须含有time_unit层级的date_index（因为查找前k个连续时间单位是依据date_index进行的）
        2. target_cols中的列最好不要存在缺失值（极端情况如某个时间段内全为缺失值的话，无法计算统计量）
    k : int
    time_unit : day/week/month/quarter/year
    target_cols : 该参数和agg_funcs参数一起决定了为哪些列计算哪些统计量
        - str:
            * str: single statistic of single column
            * iterable: multi statistics of single column
        - list/set/tuple:
            * str: single statistic of multi column
            * iterable: multi statistic of multi column
        - dict: 当target_cols传入dict类型时，agg_funcs为None，即使有输入值也不生效
            * {col1: func11, col2: iterable(func21, ...), ...}
    agg_funcs : None/str/iterable. usual functions： sum/mean/median/min/max/var
    date_index_col: time_unit层级的date_index列的列名。如无传入，则默认为f"{time_unit}_index"
    id_col : 如果没有指定，则表明传入的DataFrame是一个时间序列；
             如果指定，则按该列分组后再为每个时间节点计算目标统计量

    Returns
    -------
    res_df : pd.DateFrame.
        columns: (id_col), date_index_col,
                 f'last_{1}_discrete_{time_unit}_{col}_{func}', ... f'last_{k}_discrete_{time_unit}_{col}_{func}'

    """
    if date_index_col is None:
        date_index_col = f"{time_unit}_index"
    if date_index_col not in df.columns:
        print(f"please input date_index_col of {time_unit} level")
        return

    col_func_tuple_list = _gen_col_func_tuple_list(target_cols, agg_funcs)
    if col_func_tuple_list is None:
        return

    if id_col is None:
        grouped = [(None, df)]
    else:
        grouped = df.groupby(id_col)

    res_df_list = []
    for sku, sku_df in grouped:
        start, end = sku_df[date_index_col].min(), sku_df[date_index_col].max()
        d = dict(list(sku_df.groupby(date_index_col)))
        data = []
        for m in range(start, end + 1):
            m_df = d.get(m)
            if m_df is None:
                row = [None] * len(col_func_tuple_list)
            else:
                row = [getattr(m_df[col], func)() for col, func in col_func_tuple_list]
            data.append(row)
        cols = [f"{col}_{func}" for col, func in col_func_tuple_list]
        base_df = pd.DataFrame(data, columns=cols)
        base_df[date_index_col] = range(start, end + 1)
        k_df_list = []
        for i in range(1, k + 1):
            i_df = base_df.rename(columns={col: f'last_{i}_discrete_{time_unit}_' + col for col in cols})
            i_df[date_index_col] += i
            i_df.set_index(date_index_col, inplace=True)
            k_df_list.append(i_df)
        sub_df = pd.concat(k_df_list, axis=1).reset_index()
        if sku is not None:
            sub_df[id_col] = sku
        res_df_list.append(sub_df)

    res_df = pd.concat(res_df_list, ignore_index=True)
    return res_df


def add_last_k_day_cols():
    pass


def add_interval_k_time_unit_cols(k, time_unit):
    pass



