# -*- coding: utf-8 -*-
# @Time     : 2019/10/5 10:57
# @Author   : Run 
# @File     : __init__.py
# @Software : PyCharm


from df_helper.utils import *
from df_helper.transform import Cube


default_cube = Cube()
df2dict = default_cube.df2dict
