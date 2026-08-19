"""初始化：Koza 经典 ramped half-and-half（3.2）。

- ``full(depth)``：所有叶子位于同一深度的满树；
- ``grow(depth)``：形状可变的树（不超过指定深度）；
- ``initial_population``：对 ``min_init_depth..max_init_depth`` 每个深度桶，
  一半 full、一半 grow 随机生成，**绝不注入目标表达式**（3.2）。

所有随机选择必须使用传入的 ``rng: random.Random``（可复现性，3.3）。
"""

import random

from .config import GPConfig
from .representation import Node


def full(depth: int, config: GPConfig, rng: random.Random) -> Node:
    """生成所有叶子在同一深度的满树。

    深度 1 时叶子即终止符；深度 > 1 时每层都选函数节点，到最深层才选终止符。
    """
    if depth <= 1:
        return Node(rng.choice(config.terminal_set))
    value = rng.choice(config.function_set)
    return Node(value, [full(depth - 1, config, rng), full(depth - 1, config, rng)])


def grow(depth: int, config: GPConfig, rng: random.Random) -> Node:
    """生成不超过指定深度的形状可变树（Koza 的 grow 方法）。

    - 深度 ≤ 1 时叶子即终止符；
    - 根层（深度 ≥ 2）强制选函数节点，保证整树深度 ≥ 2，与 ramped 深度桶
      （``min_init_depth..max_init_depth``）的语义一致；
    - 其余每层从「函数集 ∪ 终止符集」中均匀选节点：选到终止符即停止生长，
      选到函数则递归生长（深度 - 1），因此树的形状与深度都可变。
    """
    if depth <= 1:
        return Node(rng.choice(config.terminal_set))
    value = rng.choice(config.function_set)
    return Node(value, [_grow_level(depth - 1, config, rng), _grow_level(depth - 1, config, rng)])


def _grow_level(depth: int, config: GPConfig, rng: random.Random) -> Node:
    """grow 的递归层：非根节点，允许终止符提前停止生长。"""
    if depth <= 1:
        return Node(rng.choice(config.terminal_set))
    pool = list(config.function_set) + list(config.terminal_set)
    value = rng.choice(pool)
    if value in config.terminal_set:
        return Node(value)
    return Node(value, [_grow_level(depth - 1, config, rng), _grow_level(depth - 1, config, rng)])


def random_tree(depth: int, config: GPConfig, rng: random.Random) -> Node:
    """按 full/grow 随机生成一棵深度不超过 ``depth`` 的树。"""
    if rng.random() < 0.5:
        return full(depth, config, rng)
    return grow(depth, config, rng)


def initial_population(config: GPConfig, rng: random.Random) -> list[Node]:
    """ramped half-and-half 生成规模恰为 ``population_size`` 的初始种群。

    对深度 2..6 的每个深度桶，各生成 ``population_size / 桶数`` 棵（一半 full、
    一半 grow）；种群规模不能被桶数整除时，把余数按顺序分配到前面的桶。
    最后打乱顺序，消除按深度分桶造成的排列偏置。
    """
    depths = list(range(config.min_init_depth, config.max_init_depth + 1))
    per_bucket = config.population_size // len(depths)
    remainder = config.population_size % len(depths)
    population = []
    for i, d in enumerate(depths):
        count = per_bucket + (1 if i < remainder else 0)
        for j in range(count):
            if j % 2 == 0:
                population.append(full(d, config, rng))
            else:
                population.append(grow(d, config, rng))
    rng.shuffle(population)
    if len(population) != config.population_size:
        raise AssertionError(
            f"初始种群规模 {len(population)} != 配置 {config.population_size}"
        )
    return population
