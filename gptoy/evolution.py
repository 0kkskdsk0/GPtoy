"""进化主循环：生成式（generational）GP + 精英保留（3.1 教学主线）。

``run_gp`` 是 notebook 调用的唯一「总控」抽象算子，内部按顺序调用：
``initialization → fitness → tournament_select → variation → elitism``，
并记录历代 best MSE 历史与首达代，供 notebook 画收敛曲线与验证（3.3）。
"""

import time
from dataclasses import dataclass, field

from .config import GPConfig
from .data import make_koza1_data
from .initialization import initial_population
from .operators import fitness
from .representation import Node
from .selection import tournament_select
from .utils import make_rng
from .variation import apply_variation


@dataclass
class RunResult:
    best_of_run: Node                # 历代最优个体
    best_mse_history: list[float]    # 每代 best MSE（长度 = 实际运行代数）
    generation_found: int | None     # 训练 MSE 首次 < mse_target 的代数（None = 未找回）
    n_generations: int               # 实际运行的代数（提前收敛则 < max_generations）
    gen0_mse: float                  # 第 0 代 best MSE（3.1：必须 > gen0_unsolved_mse）
    train_mse: float                 # best_of_run 在训练集上的 MSE（3.3）
    test_mse: float                  # best_of_run 在泛化集 [1,2] 上的 MSE（3.3）
    total_seconds: float             # 运行耗时（3.1：notebook 全程 < 10 分钟）
    seed: int | None = field(default=None)  # 本次运行使用的随机种子（溯源用）


def run_gp(config: GPConfig) -> RunResult:
    """以 ``config.seed`` 固定随机性，运行完整 GP，返回 ``RunResult``。

    内部使用 ``utils.make_rng(config.seed)`` 作为唯一随机源。
    第 0 代 = 初始种群；此后每代：评价 → 记录 best → 精英保留 → 锦标赛选父本
    → crossover/mutation/reproduction 生成下一代；训练 MSE 达标即提前收敛。
    """
    t0 = time.perf_counter()
    rng = make_rng(config.seed)
    x_train, y_train, x_test, y_test = make_koza1_data(config)

    population = initial_population(config, rng)
    best_of_run: Node | None = None
    best_mse_history: list[float] = []
    generation_found: int | None = None
    gen0_mse: float | None = None

    for gen in range(config.max_generations + 1):
        fitnesses = [fitness(ind, x_train, y_train) for ind in population]
        best_idx = min(range(len(fitnesses)), key=lambda i: (fitnesses[i], i))
        best_mse = fitnesses[best_idx]
        best_mse_history.append(best_mse)

        if best_of_run is None or best_mse < _best_mse_of(best_of_run, x_train, y_train):
            best_of_run = population[best_idx].deepcopy()
        if gen == 0:
            gen0_mse = best_mse

        if best_mse < config.mse_target:
            generation_found = gen
            break  # 提前收敛（训练 MSE 达标即停止，控制 notebook 时长）
        if gen == config.max_generations:
            break

        # ---- 生成下一代：精英保留 + 锦标赛选择 + 变异算子 ----
        keep = [population[best_idx].deepcopy() for _ in range(config.elitism_size)]
        n_parents = config.population_size - config.elitism_size
        parents = tournament_select(population, fitnesses, n_parents, config, rng)
        offspring = apply_variation(parents, config, rng)
        population = keep + offspring

    assert best_of_run is not None and gen0_mse is not None
    total_seconds = time.perf_counter() - t0
    return RunResult(
        best_of_run=best_of_run,
        best_mse_history=best_mse_history,
        generation_found=generation_found,
        n_generations=len(best_mse_history) - 1,
        gen0_mse=gen0_mse,
        train_mse=fitness(best_of_run, x_train, y_train),
        test_mse=fitness(best_of_run, x_test, y_test),
        total_seconds=total_seconds,
        seed=config.seed,
    )


def _best_mse_of(node: Node, x_train, y_train) -> float:
    """辅助：计算某棵树的训练 MSE（best-of-run 比较用）。"""
    return fitness(node, x_train, y_train)
