"""选择：锦标赛选择（``k = tournament_size``，有放回）。

- 每轮从种群中随机抽取 ``tournament_size`` 个个体（有放回），取 MSE 最小者；
- 重复 ``k`` 轮，返回 ``k`` 个父本（同一亲本可以被多次选中）；
- 实现简单且无需处理负适应度，notebook 中与适应度比例选择作对比讲解。

``tournament_select_indices`` 返回被选中个体的**下标**而不是节点对象：追踪模块
需要「父本在种群中的下标」来构造真实事件。两个函数共用同一实现、消费完全相同的
随机数序列，因此互相替换不会改变搜索轨迹。
"""

import random

from .config import GPConfig
from .representation import Node


def tournament_select_indices(
    population: list[Node],
    fitnesses: list[float],
    k: int,
    config: GPConfig,
    rng: random.Random,
) -> list[int]:
    """返回 ``k`` 个被选中父本在 ``population`` 中的下标（有放回），MSE 小者胜出。

    - 每轮独立抽取 ``config.tournament_size`` 个下标（有放回）；
    - 取其中适应度（MSE）最小者；并列时取下标最小者，保证行为确定。
    """
    if len(population) != len(fitnesses):
        raise ValueError("population 与 fitnesses 长度不一致")
    if k < 0:
        raise ValueError("k 不能为负")
    if not population:
        raise ValueError("种群为空，无法进行锦标赛选择")
    selected: list[int] = []
    for _ in range(k):
        indices = [rng.randrange(len(population)) for _ in range(config.tournament_size)]
        best = min(indices, key=lambda i: (fitnesses[i], i))
        selected.append(best)
    return selected


def tournament_select(
    population: list[Node],
    fitnesses: list[float],
    k: int,
    config: GPConfig,
    rng: random.Random,
) -> list[Node]:
    """返回 ``k`` 个被选中的父本（有放回），MSE 小者胜出。

    - ``k``：需要返回的父本数量；
    - 每轮从种群中独立抽取 ``config.tournament_size`` 个个体（有放回），
      取其中适应度（MSE）最小者；
    - MSE 相同（并列）时取下标最小者，行为确定。
    """
    indices = tournament_select_indices(population, fitnesses, k, config, rng)
    return [population[i] for i in indices]
