"""Koza-1 数据生成。

目标表达式 ``f(x)=x^4+x^3+x^2+x`` 的唯一合法出现位置：仅用于生成训练/泛化标签。
适应度（``operators.fitness``）只接收 ``(tree, x, y)``，绝不读取本模块的公式 ——
满足 Success Criteria 3.2「不能把目标公式写死在评价函数里」。
"""

import numpy as np

from .config import GPConfig


def koza1_target(x):
    """目标表达式 ``f(x) = x**4 + x**3 + x**2 + x``（仅用于生成数据标签，3.3）。

    支持标量或 ``np.ndarray``。
    """
    return x**4 + x**3 + x**2 + x


def make_koza1_data(config: GPConfig):
    """返回 ``(x_train, y_train, x_test, y_test)``。

    - 训练集：``[-1, 1]`` 等距 ``config.train_points`` 点（3.3）
    - 泛化集：``[1, 2]`` 等距 ``config.test_points`` 点（3.3 泛化验证）
    """
    x_train = np.linspace(config.train_lo, config.train_hi, config.train_points)
    y_train = koza1_target(x_train)
    x_test = np.linspace(config.test_lo, config.test_hi, config.test_points)
    y_test = koza1_target(x_test)
    return x_train, y_train, x_test, y_test
