"""进化主循环：生成式（generational）GP + 精英保留，以及可逐代调用的 ``EvolutionEngine``。

两条入口
--------
- ``EvolutionEngine(config, tracer=None)``：逐代引擎。
  ``initialize()`` 建立第 0 代；``step()`` 推进一代并返回该代 ``GenerationSnapshot``；
  ``run()`` 一直跑到停止条件并返回 ``RunResult``。这是**唯一**的算法实现。
- ``run_gp(config, tracer=None)``：内部就是 ``EvolutionEngine(...).run()``，
  因此「逐代执行」与「一次性批量执行」在相同 seed/config 下历史完全一致。

一代做什么
----------
1. 评价当前种群（训练 MSE；非有限值记为 ``inf``）；
2. 记录 best-of-generation 快照（含训练/测试 MSE、适应度分布、规模/深度、结构唯一性）；
3. 更新 best-of-run 与 Hall of Fame（严格变优时追加）；
4. 若训练 MSE < ``mse_target`` 且 ``stop_on_target=True``，立即停止；
5. 否则：精英保留 + 锦标赛选择父本 + crossover/mutation/reproduction 生成同龄下一代。

评估计数约定
------------
``evaluations`` 只统计**为搜索而做的训练适应度评估**：每评价一代种群计
``population_size`` 次。测试集评价（快照里的 ``best_test_mse``、最终
``RunResult.test_mse``）不计入预算、也不反馈给选择；使用自定义 ``fitness_fn`` 时，
为记录原始 MSE 而额外做的一次诊断评价同样不计入 ``evaluations``。
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

from .config import GPConfig
from .data import make_koza1_data
from .initialization import initial_population
from .metrics import population_stats
from .operators import fitness as default_fitness
from .representation import Node
from .selection import tournament_select_indices
from .tracing import EvolutionTracer, GenerationSnapshot, GeneticEvent, NullTracer
from .utils import make_rng
from .variation import apply_variation, apply_variation_traced


@dataclass
class RunResult:
    """一次完整运行的返回值（保留旧字段，新增观测字段）。"""

    # ---- 旧字段（字段顺序保持兼容）----
    best_of_run: Node                # 历代最优个体（按适应度）
    best_mse_history: list[float]    # 每代 best 训练 MSE（长度 = 实际评价代数）
    generation_found: int | None     # 训练 MSE 首次 < mse_target 的代数（None = 未达标）
    n_generations: int               # 实际运行的代数（提前收敛则 < max_generations）
    gen0_mse: float                  # 第 0 代 best 训练 MSE
    train_mse: float                 # best_of_run 的训练集 MSE；自定义目标全为 inf 时报告 inf（见 _make_result）
    test_mse: float                  # best_of_run 在测试/泛化集上的 MSE
    total_seconds: float             # 运行耗时
    seed: int | None = None          # 本次运行使用的随机种子

    # ---- 新增字段 ----
    evaluations: int = 0                                  # 训练适应度评估次数（搜索预算）
    snapshots: list[GenerationSnapshot] = field(default_factory=list)
    stop_reason: str = ""                                 # "converged" / "max_generations"
    config: GPConfig | None = None
    train_mse_history: list[float] = field(default_factory=list)
    test_mse_history: list[float] = field(default_factory=list)
    hall_of_fame: list[Node] = field(default_factory=list)


class EvolutionEngine:
    """逐代 GP 引擎：``initialize`` → 反复 ``step`` → ``result``。

    参数
    ----
    config : GPConfig
        全部超参数（种群、代数、深度、概率、阈值、seed、``stop_on_target``）。
    tracer : EvolutionTracer | None
        传 ``EvolutionTracer()`` 时记录逐代快照与遗传事件；``None`` 时不构造事件。
    data : tuple | dict | None
        自定义 ``(x_train, y_train, x_test, y_test)``，或含同名键的 dict
        （``make_train_val_test`` 的返回值即可直接传入）。``None`` 时使用 Koza-1 数据。
    fitness_fn : callable | None
        ``(node, x, y) -> float`` 的自定义搜索目标（如加入简约惩罚）；``None`` 时
        使用 ``operators.fitness``（训练 MSE）。自定义目标只影响选择，快照仍记录
        原始训练 MSE，测试集评价始终使用原始 MSE。
    """

    def __init__(
        self,
        config: GPConfig,
        tracer: EvolutionTracer | None = None,
        *,
        data=None,
        fitness_fn=None,
    ) -> None:
        self.config = config
        self.tracer = tracer if tracer is not None else NullTracer()
        self._trace = bool(getattr(self.tracer, "enabled", False))
        # 只保留聚合 snapshots 的实验（keep_events=False）不构造事件对象
        self._trace_events = self._trace and bool(getattr(self.tracer, "wants_events", True))
        self._validate_config()

        self.x_train, self.y_train, self.x_test, self.y_test = _coerce_data(config, data)
        self._fitness_fn = fitness_fn if fitness_fn is not None else default_fitness
        self._custom_fitness = fitness_fn is not None

        self.rng = make_rng(config.seed)
        self.population: list[Node] = []
        self.fitnesses: list[float] = []
        self.generation: int = -1
        self.done: bool = False
        self.evaluations: int = 0
        self.generation_found: int | None = None
        self.stop_reason: str = ""
        self.snapshots: list[GenerationSnapshot] = []
        self.hall_of_fame: list[Node] = []

        self._best_of_run: Node | None = None
        self._best_fitness: float = math.inf
        self._last_best_idx: int | None = None
        self._pending_events: list[tuple[GeneticEvent, tuple[int, ...]]] = []
        self._started = False
        self._result: RunResult | None = None
        self._t0 = time.perf_counter()

    # ------------------------------------------------------------------ 配置
    def _validate_config(self) -> None:
        cfg = self.config
        if cfg.population_size < 1:
            raise ValueError("population_size 必须 ≥ 1")
        if not (0 <= cfg.elitism_size <= cfg.population_size):
            raise ValueError("elitism_size 必须在 [0, population_size] 内")
        if cfg.max_generations < 0:
            raise ValueError("max_generations 必须 ≥ 0")
        if cfg.tournament_size < 1:
            raise ValueError("tournament_size 必须 ≥ 1")
        if cfg.max_tree_depth < 1:
            raise ValueError("max_tree_depth 必须 ≥ 1")
        if cfg.variation_retries < 0:
            raise ValueError("variation_retries 必须 ≥ 0")
        if not (0.0 <= cfg.p_crossover <= 1.0) or not (0.0 <= cfg.p_mutation <= 1.0):
            raise ValueError("p_crossover / p_mutation 必须在 [0, 1] 内")
        if cfg.p_crossover + cfg.p_mutation > 1.0 + 1e-12:
            raise ValueError("p_crossover + p_mutation 不能超过 1（余量为 reproduction）")
        if cfg.min_init_depth < 1 or cfg.max_init_depth < cfg.min_init_depth:
            raise ValueError("初始化深度需满足 1 ≤ min_init_depth ≤ max_init_depth")
        # 注意：这里故意不强制 max_init_depth <= max_tree_depth。
        # 正常配置应保持 init 深度不超过 max_tree_depth；但教学实验（如交叉深度回退）会故意
        # 用超深初始树触发 fallback，因此保留这一自由度，而不是在构造时直接报错。

    # ------------------------------------------------------------------ 逐代接口
    def initialize(self) -> GenerationSnapshot:
        """建立第 0 代（初始种群）、评价并记录快照。"""
        if self._started:
            return self.snapshots[-1]
        self.population = initial_population(self.config, self.rng)
        self.generation = 0
        self._started = True
        self._evaluate_generation()
        return self.snapshots[-1]

    def step(self) -> GenerationSnapshot | None:
        """从当前种群生成下一代并评价；已停止时返回 ``None``。"""
        if not self._started:
            raise RuntimeError("请先调用 initialize() 建立第 0 代")
        if self.done:
            return None

        cfg = self.config
        prev_population = self.population
        prev_fitnesses = self.fitnesses
        assert self._last_best_idx is not None
        best_idx = self._last_best_idx

        # 1) 精英保留（深拷贝，避免下一代与上一代共享节点）
        keep = [prev_population[best_idx].deepcopy() for _ in range(cfg.elitism_size)]
        n_offspring = cfg.population_size - cfg.elitism_size

        # 2) 锦标赛选择父本（下标版本；与 tournament_select 消费相同随机数）
        parent_indices = tournament_select_indices(
            prev_population, prev_fitnesses, n_offspring, cfg, self.rng
        )
        parents = [prev_population[i] for i in parent_indices]

        # 3) 变异算子（traced / 普通版本共享同一 RNG 消费序列）
        if self._trace_events:
            offspring, records = apply_variation_traced(parents, cfg, self.rng)
        else:
            offspring = apply_variation(parents, cfg, self.rng)
            records = []

        self.population = keep + offspring
        self.generation += 1

        # 4) 构造真实事件（子代适应度要等评价后才知道，先挂起）
        self._pending_events = []
        if self._trace_events:
            self._build_events(prev_population, prev_fitnesses, parent_indices, keep, records)

        # 5) 评价新一代并记录快照（内部会把挂起事件补全并交给 tracer）
        self._evaluate_generation()
        self._result = None
        return self.snapshots[-1]

    def run(self) -> RunResult:
        """跑到停止条件（或已停止）并返回 ``RunResult``。"""
        if not self._started:
            self.initialize()
        while not self.done:
            self.step()
        if self._result is None:
            self._result = self._make_result()
        return self._result

    # ------------------------------------------------------------------ 内部
    def _evaluate_generation(self) -> None:
        """评价当前种群、生成快照、更新 best-of-run 与停止条件。"""
        cfg = self.config
        fitnesses: list[float] = []
        for ind in self.population:
            value = float(self._fitness_fn(ind, self.x_train, self.y_train))
            if not math.isfinite(value):
                value = float("inf")  # 明确策略：NaN/Inf 一律视为最差
            fitnesses.append(value)
        self.fitnesses = fitnesses
        self.evaluations += len(self.population)

        best_idx = min(range(len(fitnesses)), key=lambda i: (fitnesses[i], i))
        best_fitness = fitnesses[best_idx]
        best_tree = self.population[best_idx]
        best_train_mse = (
            float(default_fitness(best_tree, self.x_train, self.y_train))
            if self._custom_fitness
            else best_fitness
        )
        best_test_mse = float(default_fitness(best_tree, self.x_test, self.y_test))

        stats = population_stats(self.population, fitnesses, self.generation)
        snapshot = GenerationSnapshot(
            generation=self.generation,
            best_expression=best_tree.sexpr_str(),
            best_train_mse=best_train_mse,
            best_test_mse=best_test_mse,
            best_fitness=best_fitness,
            mean_fitness=stats["mean_fitness"],
            median_fitness=stats["median_fitness"],
            worst_fitness=stats["worst_fitness"],
            mean_size=stats["mean_size"],
            max_size=stats["max_size"],
            mean_depth=stats["mean_depth"],
            max_depth=stats["max_depth"],
            unique_structures=stats["unique_structures"],
            unique_ratio=stats["unique_ratio"],
            evaluations=self.evaluations,
            size_histogram=stats["size_histogram"],
            depth_histogram=stats["depth_histogram"],
            fitness_std=stats["fitness_std"],
        )
        self.snapshots.append(snapshot)
        if self._trace:
            self.tracer.on_snapshot(snapshot)

        # best-of-run / Hall of Fame：只在严格变优时更新（与旧实现的 `<` 比较一致）。
        # 首代即使所有个体都是 inf（例如自定义 fitness 无有限值），也必须选定 best_of_run，
        # 否则 RunResult 无法构造；这不会改变有限适应度时的默认轨迹。
        if self._best_of_run is None or best_fitness < self._best_fitness:
            self._best_fitness = best_fitness
            self._best_of_run = best_tree.deepcopy()
            self.hall_of_fame.append(best_tree.deepcopy())

        if self.generation_found is None and best_train_mse < cfg.mse_target:
            self.generation_found = self.generation

        self._flush_events()
        self._last_best_idx = best_idx

        # 停止条件：训练 MSE 达标（可关闭）或达到代数上限
        if self.generation_found is not None and cfg.stop_on_target:
            self.done = True
            self.stop_reason = "converged"
        elif self.generation >= cfg.max_generations:
            self.done = True
            self.stop_reason = "max_generations"

    def _build_events(
        self,
        prev_population: list[Node],
        prev_fitnesses: list[float],
        parent_indices: list[int],
        keep: list[Node],
        records,
    ) -> None:
        """把精英保留与算子记录转换成 ``GeneticEvent``（子代适应度稍后补全）。"""
        generation = self.generation
        best_idx = self._last_best_idx
        assert best_idx is not None
        for offset, elite_child in enumerate(keep):
            event = GeneticEvent(
                generation=generation,
                event_type="elitism",
                parent_indices=(best_idx,),
                parent_expressions=(prev_population[best_idx].sexpr_str(),),
                child_expressions=(elite_child.sexpr_str(),),
                parent_fitnesses=(prev_fitnesses[best_idx],),
                changed=False,
                elite=True,
            )
            self._pending_events.append((event, (offset,)))

        child_offset = len(keep)
        for record in records:
            positions = tuple(range(child_offset, child_offset + len(record.children)))
            child_offset += len(record.children)
            event_parent_indices = tuple(parent_indices[p] for p in record.parent_positions)
            event = GeneticEvent(
                generation=generation,
                event_type=record.event_type,
                parent_indices=event_parent_indices,
                parent_expressions=tuple(
                    prev_population[i].sexpr_str() for i in event_parent_indices
                ),
                child_expressions=tuple(c.sexpr_str() for c in record.children),
                crossover_points=record.crossover_points,
                mutation_point=record.mutation_point,
                parent_fitnesses=tuple(prev_fitnesses[i] for i in event_parent_indices),
                changed=record.changed,
                fallback=record.fallback,
                elite=False,
            )
            self._pending_events.append((event, positions))

    def _flush_events(self) -> None:
        """用新一代的真实适应度补全挂起事件，并交给 tracer。"""
        if not self._pending_events:
            return
        for event, positions in self._pending_events:
            event.child_fitnesses = tuple(self.fitnesses[p] for p in positions)
            self.tracer.on_event(event)
        self._pending_events = []

    def _make_result(self) -> RunResult:
        if not self.snapshots or self._best_of_run is None:
            raise RuntimeError("尚未运行：请先调用 initialize()/step() 或 run()")
        cfg = self.config
        best = self._best_of_run.deepcopy()
        train_mse = float(default_fitness(best, self.x_train, self.y_train))
        if self._custom_fitness:
            # 自定义搜索目标只用于选择，train_mse 默认仍报告原始 MSE；
            # 但若该目标在 best_of_run 上仍非有限（例如所有个体 fitness=inf），
            # 说明本次运行没有任何有限目标值，此时报告 inf，而不是一个“看起来正常”的 MSE。
            objective_value = float(self._fitness_fn(best, self.x_train, self.y_train))
            if not math.isfinite(objective_value):
                train_mse = float("inf")
        return RunResult(
            best_of_run=best,
            best_mse_history=[s.best_train_mse for s in self.snapshots],
            generation_found=self.generation_found,
            n_generations=self.generation,
            gen0_mse=self.snapshots[0].best_train_mse,
            train_mse=train_mse,
            test_mse=float(default_fitness(best, self.x_test, self.y_test)),
            total_seconds=time.perf_counter() - self._t0,
            seed=cfg.seed,
            evaluations=self.evaluations,
            snapshots=list(self.snapshots),
            stop_reason=self.stop_reason or ("in_progress" if not self.done else ""),
            config=cfg,
            train_mse_history=[s.best_train_mse for s in self.snapshots],
            test_mse_history=[s.best_test_mse for s in self.snapshots],
            hall_of_fame=list(self.hall_of_fame),
        )

    # ------------------------------------------------------------------ 属性
    @property
    def best_of_run(self) -> Node | None:
        """当前 best-of-run（未运行时为 ``None``）。"""
        return self._best_of_run

    @property
    def result(self) -> RunResult:
        """当前运行结果；运行中访问会得到截至当前的快照汇总。"""
        if self._result is None:
            if not self._started:
                raise RuntimeError("尚未运行：请先调用 initialize()/step() 或 run()")
            self._result = self._make_result()
        return self._result


def _coerce_data(config: GPConfig, data):
    """把 ``data`` 参数统一成 ``(x_train, y_train, x_test, y_test)`` 四个数组。"""
    if data is None:
        return make_koza1_data(config)
    if isinstance(data, dict):
        try:
            return data["x_train"], data["y_train"], data["x_test"], data["y_test"]
        except KeyError as exc:  # 明确报错，避免用错字段
            raise ValueError(f"data dict 缺少键 {exc}") from exc
    if isinstance(data, (tuple, list)) and len(data) == 4:
        return tuple(data)
    raise ValueError("data 必须是 (x_train, y_train, x_test, y_test) 或含同名键的 dict")


def run_gp(
    config: GPConfig,
    tracer: EvolutionTracer | None = None,
    *,
    data=None,
    fitness_fn=None,
) -> RunResult:
    """以 ``config.seed`` 固定随机性，运行完整 GP（``EvolutionEngine`` 的薄封装）。

    与逐代执行完全等价::

        engine = EvolutionEngine(config)
        engine.initialize()
        while not engine.done:
            engine.step()
        result = engine.result
    """
    return EvolutionEngine(config, tracer, data=data, fitness_fn=fitness_fn).run()
