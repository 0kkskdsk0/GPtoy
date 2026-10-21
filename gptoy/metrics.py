"""种群指标：结构签名、适应度/复杂度/多样性统计。

设计原则
--------
- **结构签名唯一性**：``structural_signature(tree)`` 直接用 S-表达式，与
  ``Node.__eq__`` / ``Node.__hash__`` 完全一致（结构相同 ⇒ 签名相同）。
- **只做聚合**：``population_stats`` 只返回每代的聚合统计与直方图，不保存
  完整种群，供大规模多种子实验低成本记录（See ``gptoy.tracing``）。
- 适应度约定：本项目适应度是 MSE，**越小越好**，因此 ``best_fitness = min(...)``，
  ``worst_fitness = max(...)``。
"""

from __future__ import annotations

import math

import numpy as np

from .representation import Node


def structural_signature(tree: Node) -> str:
    """稳定、唯一的树结构签名（S-表达式）。

    与 ``Node.__eq__`` 一致：``a == b`` ⇔ ``structural_signature(a) ==
    structural_signature(b)``。因此 ``len({structural_signature(t) for t in pop})``
    就是结构唯一数。
    """
    return tree.sexpr_str()


def _histogram(values: list[int]) -> dict[int, int]:
    """把整数列表转成「取值 → 计数」的字典（键升序）。"""
    out: dict[int, int] = {}
    for v in values:
        out[int(v)] = out.get(int(v), 0) + 1
    return dict(sorted(out.items()))


def population_stats(
    population: list[Node],
    fitnesses: list[float] | None = None,
    generation: int | None = None,
) -> dict:
    """计算一代种群的适应度、规模、深度与多样性统计。

    参数
    ----
    population : list[Node]
        当代种群（不能为空）。
    fitnesses : list[float] | None
        与 ``population`` 等长的适应度（MSE，越小越好）。为 ``None`` 时
        适应度相关字段返回 ``None``；``inf`` 会被如实计入 ``worst_fitness``。
    generation : int | None
        当前代数，原样写回结果，便于把多代统计拼成表。

    返回
    ----
    dict，至少包含：
        generation, best_fitness, mean_fitness, median_fitness, worst_fitness,
        fitness_std, mean_size, max_size, mean_depth, max_depth,
        unique_structures, unique_ratio, size_histogram, depth_histogram,
        population_size

    其中 ``unique_ratio = unique_structures / population_size``；
    直方图是 ``{取值: 计数}``；``fitness_std`` 使用总体标准差（``ddof=0``）。
    """
    if not population:
        raise ValueError("population_stats 需要非空种群")
    if fitnesses is not None and len(fitnesses) != len(population):
        raise ValueError("population 与 fitnesses 长度不一致")

    sizes = [tree.size() for tree in population]
    depths = [tree.depth() for tree in population]
    signatures = [structural_signature(tree) for tree in population]
    unique_structures = len(set(signatures))

    if fitnesses is None:
        best_fitness = mean_fitness = median_fitness = worst_fitness = None  # type: ignore[assignment]
        fitness_std = None  # type: ignore[assignment]
    else:
        finite_values = np.asarray(fitnesses, dtype=float)
        best_fitness = float(np.min(finite_values))
        mean_fitness = float(np.mean(finite_values))
        median_fitness = float(np.median(finite_values))
        worst_fitness = float(np.max(finite_values))
        with np.errstate(invalid="ignore", over="ignore"):
            std = float(np.std(finite_values))  # 总体标准差 ddof=0
        # 出现 inf 时 inf - inf 会得到 nan；这里明确记为 inf（"离散度无界"）
        fitness_std = std if np.isfinite(std) else float("inf")

    return {
        "generation": generation,
        "population_size": len(population),
        "best_fitness": best_fitness,
        "mean_fitness": mean_fitness,
        "median_fitness": median_fitness,
        "worst_fitness": worst_fitness,
        "fitness_std": fitness_std,
        "mean_size": float(np.mean(sizes)),
        "max_size": int(max(sizes)),
        "mean_depth": float(np.mean(depths)),
        "max_depth": int(max(depths)),
        "unique_structures": unique_structures,
        "unique_ratio": unique_structures / len(population),
        "size_histogram": _histogram(sizes),
        "depth_histogram": _histogram(depths),
    }


def complexity(tree: Node) -> dict:
    """单个表达式树的结构复杂度指标（教学用）。"""
    counts: dict[str, int] = {}
    for node, _ in tree.iter_nodes():
        counts[node.value] = counts.get(node.value, 0) + 1
    return {
        "size": tree.size(),
        "depth": tree.depth(),
        "node_counts": dict(sorted(counts.items())),
        "sexpr": tree.sexpr_str(),
    }


def is_finite_fitness(value: float) -> bool:
    """判断适应度是否为有限值（``inf`` 用于标记数值异常个体）。"""
    return isinstance(value, (int, float)) and math.isfinite(value)
