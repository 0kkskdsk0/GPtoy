"""变异算子：subtree crossover / subtree mutation / reproduction（3.1 的可观测对象）。

- ``crossover(p1, p2)``：随机取两亲本子树交换；结果深度超过 ``max_tree_depth``
  则重试（有限次），仍超限则回退为亲本副本，保证树结构始终合法；
- ``mutate(ind)``：随机选一个子树，替换为 grow 生成的新子树（受深度约束）；
- ``reproduce(ind)``：深拷贝，保证与亲本完全独立；
- ``apply_variation``：按 ``p_crossover / p_mutation / 余量为 reproduction`` 概率分发。
"""

import random

from .config import GPConfig
from .initialization import grow
from .representation import Node


# ---- 树内部结构操作（仅本模块使用）----

def _collect(node: Node, path: tuple = ()) -> list[tuple[Node, tuple]]:
    """返回 ``[(节点, 根到该节点的路径), ...]``，路径为子节点下标元组（根路径为空）。"""
    nodes = [(node, path)]
    for i, child in enumerate(node.children):
        nodes.extend(_collect(child, path + (i,)))
    return nodes


def _subtree_at(root: Node, path: tuple) -> Node:
    """取路径指向的子树（不修改原树）。"""
    node = root
    for i in path:
        node = node.children[i]
    return node


def _replace(root: Node, path: tuple, new_subtree: Node) -> Node:
    """返回「把 root 深拷贝后、将路径处子树替换为 new_subtree 副本」的新树。"""
    if not path:
        return new_subtree.deepcopy()
    new_root = root.deepcopy()
    node = new_root
    for i in path[:-1]:
        node = node.children[i]
    node.children[path[-1]] = new_subtree.deepcopy()
    return new_root


def _random_path(node: Node, rng: random.Random) -> tuple:
    """均匀随机选一个节点并返回其路径。"""
    nodes = _collect(node)
    return rng.choice(nodes)[1]


def crossover(p1: Node, p2: Node, config: GPConfig, rng: random.Random) -> tuple[Node, Node]:
    """子树交叉，返回两个子代。

    在两个亲本上各随机选一个交叉点，交换两棵子树；若任一子代深度超过
    ``max_tree_depth`` 则重新选点，最多重试 ``variation_retries`` 次，
    仍超限则回退为两个亲本的深拷贝（树结构始终保持合法）。
    """
    for _ in range(config.variation_retries):
        path1 = _random_path(p1, rng)
        path2 = _random_path(p2, rng)
        child1 = _replace(p1, path1, _subtree_at(p2, path2))
        child2 = _replace(p2, path2, _subtree_at(p1, path1))
        if (
            child1.depth() <= config.max_tree_depth
            and child2.depth() <= config.max_tree_depth
        ):
            return child1, child2
    return p1.deepcopy(), p2.deepcopy()


def mutate(ind: Node, config: GPConfig, rng: random.Random) -> Node:
    """子树变异：随机子树替换为随机新子树。

    随机选一个节点作为变异点，把该子树替换为一棵 grow 新树；新树深度上限为
    ``max_tree_depth - 变异点深度``，保证变异后整棵树不超深。
    """
    path = _random_path(ind, rng)
    depth_budget = max(1, config.max_tree_depth - len(path))
    new_subtree = grow(depth_budget, config, rng)
    return _replace(ind, path, new_subtree)


def reproduce(ind: Node) -> Node:
    """复制（深拷贝）。"""
    return ind.deepcopy()


def apply_variation(parents: list[Node], config: GPConfig, rng: random.Random) -> list[Node]:
    """按概率分发三种算子，生成与 ``parents`` 等长的下一代个体列表。

    按顺序消费父本：``p_crossover`` 概率取相邻两个父本交叉（产生两个子代），
    ``p_mutation`` 概率对单个父本变异，其余概率直接复制。
    """
    offspring = []
    i = 0
    while i < len(parents):
        r = rng.random()
        if r < config.p_crossover and i + 1 < len(parents):
            child1, child2 = crossover(parents[i], parents[i + 1], config, rng)
            offspring.extend([child1, child2])
            i += 2
        elif r < config.p_crossover + config.p_mutation:
            offspring.append(mutate(parents[i], config, rng))
            i += 1
        else:
            offspring.append(reproduce(parents[i]))
            i += 1
    if len(offspring) != len(parents):
        raise AssertionError(
            f"apply_variation 输出规模 {len(offspring)} != 输入 {len(parents)}"
        )
    return offspring
