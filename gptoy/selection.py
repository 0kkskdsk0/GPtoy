"""选择：锦标赛选择（``k = tournament_size``，有放回）。

- 从种群中随机抽取 k 个个体，取 MSE 最小者进入下一代；
- 允许同一亲本被多次选中；
- 实现简单且无需处理负适应度，notebook 中与适应度比例选择作对比讲解。
"""

import random

from .config import GPConfig
from .representation import Node


def tournament_select(
    population: list[Node],
    fitnesses: list[float],
    k: int,
    config: GPConfig,
    rng: random.Random,
) -> list[Node]:
    """返回 k 个被选中的父本（有放回），MSE 小者胜出。

    - ``k``：需要返回的父本数量；
    - 每轮从种群中独立抽取 ``config.tournament_size`` 个个体（有放回），
      取其中适应度（MSE）最小者；
    - MSE 相同（并列）时取下标最小者，行为确定。
    """
    if len(population) != len(fitnesses):
        raise ValueError("population 与 fitnesses 长度不一致")
    if k < 0:
        raise ValueError("k 不能为负")
    parents = []
    for _ in range(k):
        indices = [rng.randrange(len(population)) for _ in range(config.tournament_size)]
        best = min(indices, key=lambda i: (fitnesses[i], i))
        parents.append(population[best])
    return parents
