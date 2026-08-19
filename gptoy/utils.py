"""工具：单一随机源与可复现性保证（3.3）。

全项目所有随机操作（初始化/选择/变异）都必须使用 ``make_rng(seed)`` 返回的
``random.Random`` 实例，保证同一 seed 得到逐位一致的运行结果。
禁止在算子实现中散落裸 ``random``/``np.random`` 调用。
"""

import random

import numpy as np


def make_rng(seed: int | None) -> random.Random:
    """创建项目的唯一随机源；``seed=None`` 时使用系统熵（不保证可复现）。"""
    # random.Random(None) 使用系统熵；random.Random(seed) 的序列在 CPython 中逐位确定。
    return random.Random(seed)


def set_global_seeds(seed: int) -> None:
    """防御性固定 ``random`` 与 ``numpy.random`` 的全局种子（3.1/3.3）。

    项目的算子本身不依赖全局随机源（全部通过 ``make_rng`` 注入），
    此函数只用于防御 notebook / 第三方库中的散落随机调用。
    """
    random.seed(seed)
    np.random.seed(seed)
