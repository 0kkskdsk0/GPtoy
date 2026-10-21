"""变异算子：subtree crossover / subtree mutation / reproduction。

- ``crossover(p1, p2)``：随机取两亲本子树交换；结果深度超过 ``max_tree_depth``
  则重试（有限次），仍超限则回退为亲本副本，保证树结构始终合法；
- ``mutate(ind)``：随机选一个子树，替换为 grow 生成的新子树（受深度预算约束）；
- ``reproduce(ind)``：深拷贝，保证与亲本完全独立；
- ``apply_variation``：按 ``p_crossover / p_mutation / 余量为 reproduction`` 概率分发。

可观测接口（不改变随机数消费顺序）
----------------------------------
``crossover_traced`` / ``mutate_traced`` / ``apply_variation_traced`` 与对应的
普通接口共用同一份内部实现，因此**是否记录追踪信息不会改变随机数消费顺序**：
- ``CrossoverTrace`` 记录两个交叉点路径、两个子代与是否发生深度回退；
- ``MutationTrace`` 记录变异点路径、新子树深度与子代；
- ``VariationRecord`` 记录一次算子分发：事件类型、父本在输入列表中的位置、
  子代对象、交叉点/变异点、是否回退、结构是否真的改变。

路径是「从根出发的子节点下标元组」，空元组代表根节点（见 ``representation.Node``）。
"""

import random
from dataclasses import dataclass

from .config import GPConfig
from .initialization import grow
from .representation import Node

# ---- 树内部结构操作（复用 representation 的路径接口）----


def _random_path(node: Node, rng: random.Random) -> tuple[int, ...]:
    """均匀随机选一个节点并返回其路径（先序遍历顺序固定）。"""
    return rng.choice(list(node.iter_nodes()))[1]


# ---- 追踪记录（dataclass，便于 notebook 直接访问属性）----


@dataclass(frozen=True)
class CrossoverTrace:
    """一次真实子树交叉的记录。"""

    child1: Node
    child2: Node
    point1: tuple[int, ...]
    point2: tuple[int, ...]
    fallback: bool = False


@dataclass(frozen=True)
class MutationTrace:
    """一次真实子树变异的记录。"""

    child: Node
    point: tuple[int, ...]
    new_subtree_depth: int
    fallback: bool = False


@dataclass(frozen=True)
class VariationRecord:
    """``apply_variation`` 中一次算子分发的记录（供进化引擎构造 GeneticEvent）。"""

    event_type: str                       # "crossover" / "mutation" / "reproduction"
    parent_positions: tuple[int, ...]     # 父本在 apply_variation 输入列表中的位置
    children: tuple[Node, ...]            # 本次产生的子代
    crossover_points: tuple[tuple[int, ...], tuple[int, ...]] | None = None
    mutation_point: tuple[int, ...] | None = None
    fallback: bool = False
    changed: bool = False


# ---- 核心实现（trace / no-trace 共用，随机数消费完全一致）----


def _crossover_core(
    p1: Node, p2: Node, config: GPConfig, rng: random.Random
) -> tuple[Node, Node, tuple[int, ...], tuple[int, ...], bool]:
    """交叉核心：返回 ``(child1, child2, point1, point2, fallback)``。"""
    point1: tuple[int, ...] = ()
    point2: tuple[int, ...] = ()
    for _ in range(config.variation_retries):
        point1 = _random_path(p1, rng)
        point2 = _random_path(p2, rng)
        child1 = p1.replace_subtree(point1, p2.subtree_at(point2))
        child2 = p2.replace_subtree(point2, p1.subtree_at(point1))
        if (
            child1.depth() <= config.max_tree_depth
            and child2.depth() <= config.max_tree_depth
        ):
            return child1, child2, point1, point2, False
    # 有限次重试后仍超深：回退为亲本副本（保留最后一次尝试的交叉点供观察）
    return p1.deepcopy(), p2.deepcopy(), point1, point2, True


def _mutate_core(
    ind: Node, config: GPConfig, rng: random.Random
) -> tuple[Node, tuple[int, ...], int]:
    """变异核心：返回 ``(child, point, new_subtree_depth)``。"""
    point = _random_path(ind, rng)
    depth_budget = max(1, config.max_tree_depth - len(point))
    new_subtree = grow(depth_budget, config, rng)
    return ind.replace_subtree(point, new_subtree), point, new_subtree.depth()


def crossover(p1: Node, p2: Node, config: GPConfig, rng: random.Random) -> tuple[Node, Node]:
    """子树交叉，返回两个子代。

    在两个亲本上各随机选一个交叉点，交换两棵子树；若任一子代深度超过
    ``max_tree_depth`` 则重新选点，最多重试 ``variation_retries`` 次，
    仍超限则回退为两个亲本的深拷贝（树结构始终保持合法）。
    """
    child1, child2, _, _, _ = _crossover_core(p1, p2, config, rng)
    return child1, child2


def crossover_traced(
    p1: Node, p2: Node, config: GPConfig, rng: random.Random
) -> CrossoverTrace:
    """与 ``crossover`` 完全相同的随机过程，但返回交叉点/回退记录。"""
    child1, child2, point1, point2, fallback = _crossover_core(p1, p2, config, rng)
    return CrossoverTrace(child1, child2, point1, point2, fallback)


def mutate(ind: Node, config: GPConfig, rng: random.Random) -> Node:
    """子树变异：随机子树替换为随机新子树。

    随机选一个节点作为变异点，把该子树替换为一棵 grow 新树；新树深度上限为
    ``max_tree_depth - 变异点深度``，保证变异后整棵树不超深。
    """
    child, _, _ = _mutate_core(ind, config, rng)
    return child


def mutate_traced(ind: Node, config: GPConfig, rng: random.Random) -> MutationTrace:
    """与 ``mutate`` 完全相同的随机过程，但返回变异点记录。"""
    child, point, new_depth = _mutate_core(ind, config, rng)
    return MutationTrace(child, point, new_depth)


def reproduce(ind: Node) -> Node:
    """复制（深拷贝）。"""
    return ind.deepcopy()


def _apply_variation(
    parents: list[Node], config: GPConfig, rng: random.Random, want_trace: bool
) -> tuple[list[Node], list[VariationRecord]]:
    """按概率分发三种算子；``want_trace`` 决定是否构造 ``VariationRecord``。

    按顺序消费父本：``p_crossover`` 概率取相邻两个父本交叉（产生两个子代），
    ``p_mutation`` 概率对单个父本变异，其余概率直接复制。
    """
    offspring: list[Node] = []
    records: list[VariationRecord] = []
    i = 0
    n = len(parents)
    while i < n:
        r = rng.random()
        # 若最后剩下一个未配对父本且抽到交叉，`i + 1 < n` 不成立，会落到下面的 mutation /
        # reproduction 分支：这是基线行为，保证输出规模守恒，且不额外消费随机数。
        if r < config.p_crossover and i + 1 < n:
            child1, child2, point1, point2, fallback = _crossover_core(
                parents[i], parents[i + 1], config, rng
            )
            offspring.extend([child1, child2])
            if want_trace:
                records.append(
                    VariationRecord(
                        event_type="crossover",
                        parent_positions=(i, i + 1),
                        children=(child1, child2),
                        crossover_points=(point1, point2),
                        fallback=fallback,
                        changed=child1 != parents[i] or child2 != parents[i + 1],
                    )
                )
            i += 2
        elif r < config.p_crossover + config.p_mutation:
            child, point, _ = _mutate_core(parents[i], config, rng)
            offspring.append(child)
            if want_trace:
                records.append(
                    VariationRecord(
                        event_type="mutation",
                        parent_positions=(i,),
                        children=(child,),
                        mutation_point=point,
                        changed=child != parents[i],
                    )
                )
            i += 1
        else:
            child = reproduce(parents[i])
            offspring.append(child)
            if want_trace:
                records.append(
                    VariationRecord(
                        event_type="reproduction",
                        parent_positions=(i,),
                        children=(child,),
                        changed=False,
                    )
                )
            i += 1
    if len(offspring) != len(parents):
        raise AssertionError(
            f"apply_variation 输出规模 {len(offspring)} != 输入 {len(parents)}"
        )
    return offspring, records


def apply_variation(parents: list[Node], config: GPConfig, rng: random.Random) -> list[Node]:
    """按概率分发三种算子，生成与 ``parents`` 等长的下一代个体列表。"""
    offspring, _ = _apply_variation(parents, config, rng, want_trace=False)
    return offspring


def apply_variation_traced(
    parents: list[Node], config: GPConfig, rng: random.Random
) -> tuple[list[Node], list[VariationRecord]]:
    """``apply_variation`` 的可观测版本：额外返回每个子代由哪种算子产生。

    随机数消费顺序与 ``apply_variation`` 完全相同；同 seed 下两者产出的子代
    逐位一致（见 ``tests/test_variation.py``）。
    """
    return _apply_variation(parents, config, rng, want_trace=True)
