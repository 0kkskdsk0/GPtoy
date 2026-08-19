"""GPtoy：从零实现的遗传编程教学包（Koza-1 问题）。

包内模块按 GP 执行阶段拆分：
  data → representation → operators → initialization → selection → variation → evolution

本包不依赖任何现成遗传编程库；所有随机性统一由 ``utils.make_rng`` 注入，
以保证固定随机种子（Success Criteria 3.3）下的可复现性。
"""

from . import (
    config,
    data,
    evolution,
    initialization,
    operators,
    representation,
    selection,
    utils,
    variation,
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
    "utils",
]
