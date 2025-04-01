#!/usr/bin/env python
# coding:utf-8
"""
@author: nivic ybyang7
@license: Apache Licence
@file: group_indics.py
@time: 2025/04/02
@contact: ybyang7@iflytek.com
@site:
@software: PyCharm

# code is far away from bugs with the god animal protecting
    I love animals. They taste delicious.
              ┏┓      ┏┓
            ┏┛┻━━━┛┻┓
            ┃      ☃      ┃
            ┃  ┳┛  ┗┳  ┃
            ┃      ┻      ┃
            ┗━┓      ┏━┛
                ┃      ┗━━━┓
                ┃  神兽保佑    ┣┓
                ┃　永无BUG！   ┏┛
                ┗┓┓┏━┳┓┏┛
                  ┃┫┫  ┃┫┫
                  ┗┻┛  ┗┻┛
"""

#  Copyright (c) 2022. Lorem ipsum dolor sit amet, consectetur adipiscing elit.
#  Morbi non lorem porttitor neque feugiat blandit. Ut vitae ipsum eget quam lacinia accumsan.
#  Etiam sed turpis ac ipsum condimentum fringilla. Maecenas magna.
#  Proin dapibus sapien vel ante. Aliquam erat volutpat. Pellentesque sagittis ligula eget metus.
#  Vestibulum commodo. Ut rhoncus gravida arcu.
import numpy as np

def group_by_continuity_numpy(arr1, arr2):
    arr1 = np.array(arr1)
    arr2 = np.array(arr2)

    # 找到不连续的索引点（相邻元素差值不等于1）
    split_indices = np.where(np.diff(arr1) != 1)[0] + 1

    # 用 split_indices 切分数组
    grouped_arr1 = np.split(arr1, split_indices)
    grouped_arr2 = np.split(arr2, split_indices)

    return [list(g) for g in grouped_arr1], [list(g) for g in grouped_arr2]


def groups_by_continuity_numpy(arr1):
    arr1 = np.array(arr1)

    # 找到不连续的索引点（相邻元素差值不等于1）
    split_indices = np.where(np.diff(arr1) != 1)[0] + 1

    # 用 split_indices 切分数组
    grouped_arr1 = np.split(arr1, split_indices)

    return [list(g) for g in grouped_arr1]

if __name__ == '__main__':
    a = [1,9,3,4,5,6,7]
    b = [2, 2, 3, 4, 5, 6, 7]
    print(group_by_continuity_numpy(a,b))
    print(groups_by_continuity_numpy(a))
