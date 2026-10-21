"""进化追踪：逐代快照（``GenerationSnapshot``）、遗传事件（``GeneticEvent``）与 tracer。

两条记录线
----------
1. **聚合快照**：每代一条 ``GenerationSnapshot``，只保存聚合统计，不保存完整种群。
   大规模多种子实验可以直接用这一层（``gptoy.experiments.save_experiment``）。
2. **遗传事件**：小型教学实验打开 ``EvolutionTracer`` 时，额外记录每个子代由哪种
   算子、从哪些父本、在哪个交叉/变异点产生，以及是否发生深度回退。

``NullTracer``（或引擎里 ``tracer=None``）不构造任何事件对象；tracer 的开/关只影响
记录，不影响 RNG 消费顺序，因此同 seed 下最终结果逐位一致（见 ``tests/test_tracing.py``）。

事件字段约定
------------
- ``generation``：**子代所属**的代数（即从 g-1 代产生 g 代个体的那次操作，记为 g）；
- ``parent_indices``：父本在**上一代种群**中的下标；
- ``crossover_points`` / ``mutation_point``：根到交叉点/变异点的路径元组，无则为 ``None``；
- ``parent_fitnesses`` / ``child_fitnesses``：上一代父本与新一代子代的 MSE；
- ``changed``：该操作是否真的改变了结构（``reproduction`` / 精英复制恒为 False）；
- ``fallback``：交叉因深度限制重试耗尽而回退为父本副本时为 True；
- ``elite``：精英保留事件为 True。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class GenerationSnapshot:
    """一代种群的聚合快照（不包含完整种群）。"""

    generation: int
    best_expression: str
    best_train_mse: float
    best_test_mse: float
    best_fitness: float
    mean_fitness: float
    median_fitness: float
    worst_fitness: float
    mean_size: float
    max_size: int
    mean_depth: float
    max_depth: int
    unique_structures: int
    unique_ratio: float
    evaluations: int
    size_histogram: dict[int, int]
    depth_histogram: dict[int, int]
    # 可选的额外统计（契约要求之外的补充，不参与任何选择）
    fitness_std: float | None = None


@dataclass
class GeneticEvent:
    """一次真实的遗传操作记录（交叉 / 变异 / 复制 / 精英保留）。"""

    generation: int
    event_type: str  # "crossover" / "mutation" / "reproduction" / "elitism"
    parent_indices: tuple[int, ...]
    parent_expressions: tuple[str, ...]
    child_expressions: tuple[str, ...]
    crossover_points: tuple[tuple[int, ...], tuple[int, ...]] | None = None
    mutation_point: tuple[int, ...] | None = None
    parent_fitnesses: tuple[float, ...] | None = None
    child_fitnesses: tuple[float, ...] | None = None
    changed: bool = False
    fallback: bool = False
    elite: bool = False

    @property
    def event_type_cn(self) -> str:
        """事件类型的中文名（notebook 展示用）。"""
        return {
            "crossover": "子树交叉",
            "mutation": "子树变异",
            "reproduction": "复制",
            "elitism": "精英保留",
        }.get(self.event_type, self.event_type)


class EvolutionTracer:
    """记录 snapshots 与 events 的 tracer。

    参数
    ----
    keep_snapshots : bool
        是否保存逐代快照（默认 True）。
    keep_events : bool
        是否保存遗传事件（默认 True；大规模实验可只保留快照以减少内存）。
    max_events : int | None
        事件上限，超过后不再追加（防止长时间运行把内存吃满）。
    """

    enabled = True
    # 引擎据此决定是否构造遗传事件；只保留 snapshots 的大规模实验可设为 False
    wants_events = True

    def __init__(
        self,
        keep_snapshots: bool = True,
        keep_events: bool = True,
        max_events: int | None = None,
    ) -> None:
        self.keep_snapshots = keep_snapshots
        self.keep_events = keep_events
        self.wants_events = bool(keep_events)
        self.max_events = max_events
        self.snapshots: list[GenerationSnapshot] = []
        self.events: list[GeneticEvent] = []
        self.dropped_events = 0

    # ---- 回调 ----
    def on_snapshot(self, snapshot: GenerationSnapshot) -> None:
        if self.keep_snapshots:
            self.snapshots.append(snapshot)

    def on_event(self, event: GeneticEvent) -> None:
        if not self.keep_events:
            return
        if self.max_events is not None and len(self.events) >= self.max_events:
            self.dropped_events += 1
            return
        self.events.append(event)

    # ---- 查询辅助 ----
    @property
    def latest_snapshot(self) -> GenerationSnapshot | None:
        return self.snapshots[-1] if self.snapshots else None

    def events_of_type(self, event_type: str) -> list[GeneticEvent]:
        """按事件类型过滤（如 ``"crossover"``、``"elitism"``）。"""
        return [e for e in self.events if e.event_type == event_type]

    def events_of_generation(self, generation: int) -> list[GeneticEvent]:
        """取某一代（子代所属代）的全部事件。"""
        return [e for e in self.events if e.generation == generation]

    def counts_by_type(self) -> dict[str, int]:
        """各事件类型的计数（教学表格用）。"""
        counts: dict[str, int] = {}
        for event in self.events:
            counts[event.event_type] = counts.get(event.event_type, 0) + 1
        return counts

    def reset(self) -> None:
        """清空已记录内容（tracer 可复用）。"""
        self.snapshots.clear()
        self.events.clear()
        self.dropped_events = 0


class NullTracer:
    """不做任何记录的 tracer：引擎在 ``tracer=None`` 时使用，事件构造被完全跳过。"""

    enabled = False
    wants_events = False

    def __init__(self) -> None:
        self.snapshots: list[GenerationSnapshot] = []
        self.events: list[GeneticEvent] = []

    def on_snapshot(self, snapshot: GenerationSnapshot) -> None:  # noqa: ARG002
        return None

    def on_event(self, event: GeneticEvent) -> None:  # noqa: ARG002
        return None

    def reset(self) -> None:
        return None
