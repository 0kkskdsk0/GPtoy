"""数据生成。

- Koza-1 目标表达式 ``f(x)=x^4+x^3+x^2+x`` 的唯一合法出现位置：仅用于生成
  训练/测试标签。适应度（``operators.fitness``）只接收 ``(tree, x, y)``，
  绝不读取本模块的公式 —— 满足「不能把目标公式写死在评价函数里」。
- ``make_target_data``：给实验（如 E10 的 sin 目标）提供通用的「目标函数 →
  数据集」入口。它只负责在给定区间上采样并计算标签，不做任何搜索或筛选。
"""

import numpy as np

from .config import GPConfig


def koza1_target(x):
    """目标表达式 ``f(x) = x**4 + x**3 + x**2 + x``（仅用于生成数据标签）。

    支持标量或 ``np.ndarray``。
    """
    return x**4 + x**3 + x**2 + x


def make_koza1_data(config: GPConfig):
    """返回 ``(x_train, y_train, x_test, y_test)``。

    - 训练集：``[-1, 1]`` 等距 ``config.train_points`` 点
    - 泛化/测试集：``[1, 2]`` 等距 ``config.test_points`` 点
    """
    x_train = np.linspace(config.train_lo, config.train_hi, config.train_points)
    y_train = koza1_target(x_train)
    x_test = np.linspace(config.test_lo, config.test_hi, config.test_points)
    y_test = koza1_target(x_test)
    return x_train, y_train, x_test, y_test


def make_target_data(
    target_fn,
    *,
    n_train: int = 20,
    train_range: tuple[float, float] = (-1.0, 1.0),
    n_test: int = 10,
    test_range: tuple[float, float] = (1.0, 2.0),
):
    """在给定区间上等距采样，用 ``target_fn`` 生成 ``(x_train, y_train, x_test, y_test)``。

    只做「采样 + 打标签」，目标函数本身不参与搜索。用于 E10 等「目标与算子集
    不匹配」的实验（例如 ``target_fn = np.sin``）。
    """
    x_train = np.linspace(train_range[0], train_range[1], n_train)
    y_train = np.asarray(target_fn(x_train), dtype=float)
    x_test = np.linspace(test_range[0], test_range[1], n_test)
    y_test = np.asarray(target_fn(x_test), dtype=float)
    return x_train, y_train, x_test, y_test
