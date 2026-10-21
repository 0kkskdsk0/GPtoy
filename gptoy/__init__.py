"""GPtoy：从零实现的遗传编程教学包（Koza-1 问题）。

包内模块按 GP 执行阶段拆分：
  data → representation → operators → initialization → selection → variation → evolution

在此之上新增三个「观察与实验」层：
  metrics      —— 结构签名与种群统计（适应度/复杂度/多样性）
  tracing      —— 逐代快照 GenerationSnapshot、遗传事件 GeneticEvent、EvolutionTracer
  experiments  —— 多种子批量、数据划分/噪声、控制变量实验、实验保存

本包不依赖任何现成遗传编程库；所有随机性统一由 ``utils.make_rng`` 注入，
以保证固定随机种子下的逐位可复现性。
"""

from . import (
    config,
    data,
    evolution,
    experiments,
    initialization,
    metrics,
    operators,
    representation,
    selection,
    tracing,
    utils,
    variation,
    viz,
)

__all__ = [
    "config",
    "data",
    "representation",
    "operators",
    "initialization",
    "selection",
    "variation",
    "evolution",
    "metrics",
    "tracing",
    "experiments",
    "utils",
    "viz",
]
