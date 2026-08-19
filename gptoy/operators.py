"""算子：保护四则运算、树递归求值、适应度（MSE）。

- 保护除法：除数为 0 时返回 1.0（Koza 经典约定），杜绝 NaN/Inf 污染适应度；
- ``evaluate``：按树后序递归求值（3.2 要求「适应度通过对树递归求值得到」）；
  ``x`` 支持标量或 ndarray，对整批训练点向量化以控制 notebook 运行时长（3.1）；
- ``fitness``：仅接收 ``(tree, x, y)``，内部不出现目标公式（3.2 红线）。
"""

import numpy as np

from .representation import Node

# 运算符 → 保护运算分派表。
_DISPATCH = {}


def _register(name):
    def deco(fn):
        _DISPATCH[name] = fn
        return fn

    return deco


@_register("+")
def p_add(a, b):
    """保护加法。"""
    return a + b


@_register("-")
def p_sub(a, b):
    """保护减法。"""
    return a - b


@_register("*")
def p_mul(a, b):
    """保护乘法。"""
    return a * b


@_register("/")
def p_div(a, b):
    """保护除法：除数（逐元素）为 0 时返回 1.0（Koza 经典约定），其余为 a / b。

    用 ``np.where(b == 0, 1.0, a / b)`` 实现：除零位置直接取 1.0，
    配合 ``errstate`` 压制除零警告，结果中绝不出现 NaN / Inf。
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        result = np.where(b == 0.0, 1.0, a / b)
    return result.item() if result.ndim == 0 else result


def evaluate(node: Node, x):
    """递归求值：函数节点先求子节点，再应用对应保护运算；终止符 ``"x"`` 返回 x 本身。"""
    if node.value == "x":
        return x
    a = evaluate(node.children[0], x)
    b = evaluate(node.children[1], x)
    return _DISPATCH[node.value](a, b)


def fitness(node: Node, x_train, y_train) -> float:
    """训练集均方误差 MSE（越小越优）。

    注意：只吃 ``(tree, x, y)``，内部绝不出现目标公式（3.2）。
    """
    return float(np.mean((evaluate(node, x_train) - y_train) ** 2))
