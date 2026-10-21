"""算子：保护四则运算、树递归求值、适应度（MSE）。

- 保护除法（``p_div``）：除数（逐元素）为 0 时返回 1.0（Koza 经典约定）。
  这是**改变数学语义**的约定：``(/ x x)`` 在 x=0 处返回 1.0 而不是未定义，
  因此不能把普通代数化简 ``x/x = 1`` 当作在全部输入上都成立的等价变换。
- ``evaluate``：按树后序递归求值；``x`` 支持标量或 ndarray。递归过程包在
  ``numpy.errstate`` 中，溢出/无效运算不会打印 warning（结果仍可能是 inf/nan）。
- ``fitness``：只接收 ``(tree, x, y)``，内部不出现目标公式。
  **非有限值策略**：若预测含 inf/nan（例如极深树在 x∈[1,2] 上溢出），
  直接返回 ``float("inf")``，即视为最差个体；绝不让 NaN 进入选择比较。

非有限值策略对原有 GP 行为的影响
--------------------------------
默认配置（种群 100、深度上限 17、seed=3）在 50 代内不会触发该分支，搜索轨迹
逐位不变；只有在出现数值溢出/NaN 的病态个体时，本策略才会把它们固定为 ``inf``
（旧实现可能得到 NaN 并污染 ``min`` 比较）。这是**有意的、被测试覆盖的行为修复**，
见 ``tests/test_operators.py`` 与 README 的「数值异常策略」一节。
"""

import numpy as np

from .representation import Node

# 非有限预测的适应度惩罚值（明确策略，而不是静默 NaN）。
NON_FINITE_FITNESS = float("inf")

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

    只把**精确等于 0** 的除数替换为 1.0；接近 0 的除数仍做普通除法（可能得到
    很大的数）。用 ``np.where(b == 0, 1.0, a / b)`` 实现，配合 ``errstate``
    压制除零警告，结果中不出现除零导致的 NaN / Inf。
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        result = np.where(b == 0.0, 1.0, a / b)
    return result.item() if result.ndim == 0 else result


def _evaluate(node: Node, x):
    """``evaluate`` 的递归内核（不做 numpy 错误状态包装，避免每层都开关 ctx）。"""
    if node.value == "x":
        return x
    a = _evaluate(node.children[0], x)
    b = _evaluate(node.children[1], x)
    return _DISPATCH[node.value](a, b)


def evaluate(node: Node, x):
    """递归求值：函数节点先求子节点，再应用对应保护运算；终止符 ``"x"`` 返回 x 本身。

    ``x`` 可为 Python 标量或 ``np.ndarray``；返回标量或与 ``x`` 同形状的数组。
    极深树可能溢出为 inf，本函数不隐藏这一事实（``fitness`` 会把非有限值判为最差）。
    """
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        return _evaluate(node, x)


def fitness(node: Node, x_train, y_train) -> float:
    """训练集均方误差 MSE（越小越优）。

    注意：

    1. 只吃 ``(tree, x, y)``，内部绝不出现目标公式；
    2. 预测含非有限值（inf/nan）或 MSE 本身非有限时返回 ``NON_FINITE_FITNESS``
       （``inf``），保证选择阶段不会出现 NaN 比较。
    """
    pred = np.asarray(evaluate(node, x_train), dtype=float)
    if not np.all(np.isfinite(pred)):
        return NON_FINITE_FITNESS
    residual = pred - np.asarray(y_train, dtype=float)
    with np.errstate(over="ignore", invalid="ignore"):
        mse = float(np.mean(residual * residual))
    return mse if np.isfinite(mse) else NON_FINITE_FITNESS
