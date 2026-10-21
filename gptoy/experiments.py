"""可复用的控制变量实验模块。

这一层只做「实验编排 + 数据生成 + 结果汇总/保存」，不复制 GP 算法：
实验对象仍然是 ``gptoy.evolution`` 的同一个引擎，因此实验里的每一代都能被
``EvolutionTracer`` 观察。

诚实性约定
----------
- ``run_seed_batch`` **保留所有种子结果**（包括失败种子），绝不按测试集筛选；
- ``summarize_runs`` 把「训练达标率」「测试达标率」「结构恢复代理率」分开报告：
  结构恢复默认是 ``None``（没有可靠的全域结构等价判定器），只有外部显式给出
  ``strict_expression``（精确树匹配，可靠但**不完备**）时才计算；
- 测试集只用于最终评价：``EvolutionEngine`` 不接受测试集参与选择，本模块也不会
  用测试 MSE 过滤或排序结果；
- 噪声公式、ddof、随机源均显式记录，保证可复现。
"""

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass, replace
from datetime import datetime
from pathlib import Path

import numpy as np

from .config import GPConfig
from .data import koza1_target
from .evolution import RunResult, run_gp
from .initialization import INIT_METHODS, initial_population
from .metrics import population_stats, structural_signature
from .operators import fitness
from .selection import tournament_select_indices
from .utils import make_rng


# ---------------------------------------------------------------------------
# 1) 多种子批量运行
# ---------------------------------------------------------------------------
def run_seed_batch(config: GPConfig, seeds, *, verbose: bool = False) -> list[RunResult]:
    """对每个种子运行一次完整 GP，返回与 ``seeds`` 等长、同顺序的结果列表。

    只替换 ``config.seed``，其余配置完全一致（控制变量）。返回列表**不做任何过滤**：
    失败的种子也保留，供统计成功率、失败案例和首达代分布。
    """
    results: list[RunResult] = []
    for seed in seeds:
        cfg = replace(config, seed=int(seed))
        result = run_gp(cfg)
        results.append(result)
        if verbose:
            hit = result.generation_found
            print(
                f"seed={seed:4d} gen0={result.gen0_mse:.4g} "
                f"hit={hit} train={result.train_mse:.3e} test={result.test_mse:.3e} "
                f"gens={result.n_generations} evals={result.evaluations}"
            )
    return results


def _mse_target_of(results: list[RunResult]) -> float:
    for result in results:
        if result.config is not None:
            return float(result.config.mse_target)
    return float(GPConfig().mse_target)


def _distribution(values) -> dict:
    """把一组数值压成 ``{count, min, median, max}``（空集时值为 None）。"""
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        return {"count": 0, "min": None, "median": None, "max": None}
    return {
        "count": int(arr.size),
        "min": float(np.min(arr)),
        "median": float(np.median(arr)),
        "max": float(np.max(arr)),
    }


def summarize_runs(
    results: list[RunResult],
    *,
    strict_expression: str | None = None,
    reference_fn=None,
    proxy_grid: tuple[float, float] = (-1.0, 2.0),
    proxy_tolerance: float = 1e-6,
) -> dict:
    """汇总多次运行：训练/测试达标率、首达代、评估次数、结构恢复代理。

    参数
    ----
    strict_expression : str | None
        已知目标树的 S-表达式。给出时，``structural_recovery_rate`` = 最优个体
        **与该树结构完全相同**的运行比例。精确树匹配是可靠的（匹配即结构恢复），
        但**不完备**：数学等价但结构不同的树会被计为未恢复，因此该比例是下界。
        未给出时为 ``None``，``structural_recovery_note`` 说明原因。
    reference_fn : callable | None
        已知目标函数。给出时额外计算 ``functional_agreement_proxy_rate``：
        在一条独立的稠密网格上 ``|pred - target| <= proxy_tolerance`` 的运行比例。
        这是**数值代理**，不是结构恢复的证明。

    返回 dict 中训练达标率与测试达标率是分开的；测试集不参与任何筛选。
    """
    if not results:
        raise ValueError("summarize_runs 需要至少一次运行结果")
    target = _mse_target_of(results)
    n = len(results)
    train_hits = [r for r in results if r.train_mse < target]
    test_hits = [r for r in results if r.test_mse < target]
    first_generations = [r.generation_found for r in results if r.generation_found is not None]

    summary: dict = {
        "n_runs": n,
        "seeds": [r.seed for r in results],
        "mse_target": target,
        "train_success_count": len(train_hits),
        "train_success_rate": len(train_hits) / n,
        "test_success_count": len(test_hits),
        "test_success_rate": len(test_hits) / n,
        "first_hit_generation": {
            **_distribution(first_generations),
            "values": [int(g) for g in first_generations],
        },
        "evaluations": {
            "total": int(sum(r.evaluations for r in results)),
            "mean": float(np.mean([r.evaluations for r in results])),
            "min": int(min(r.evaluations for r in results)),
            "max": int(max(r.evaluations for r in results)),
        },
        "train_mse": _distribution([r.train_mse for r in results]),
        "test_mse": _distribution([r.test_mse for r in results]),
        "generations": _distribution([r.n_generations for r in results]),
        "stopped_by": {
            reason: sum(1 for r in results if r.stop_reason == reason)
            for reason in sorted({r.stop_reason for r in results})
        },
        "failed_train_seeds": [r.seed for r in results if r.train_mse >= target],
        "failed_test_seeds": [r.seed for r in results if r.test_mse >= target],
        # 严格结构恢复：默认不可用（没有可靠的全域结构等价判定器）
        "structural_recovery_rate": None,
        "structural_recovery_count": None,
        "structural_recovery_available": False,
        "structural_recovery_note": (
            "unavailable：没有可靠的全域结构等价判定器；"
            "仅当显式提供 strict_expression（精确树匹配）时启用，且该比例是下界。"
        ),
        "functional_agreement_proxy_rate": None,
        "functional_agreement_proxy_note": "unavailable：未提供 reference_fn",
    }

    if strict_expression is not None:
        matches = [r for r in results if r.best_of_run.sexpr_str() == strict_expression]
        summary.update(
            {
                "structural_recovery_rate": len(matches) / n,
                "structural_recovery_count": len(matches),
                "structural_recovery_available": True,
                "structural_recovery_note": (
                    "strict_expression 精确树匹配：匹配即可确认结构恢复，"
                    "但数学等价、结构不同的树会被计为未恢复，因此该比例是下界。"
                ),
                "structurally_recovered_seeds": [r.seed for r in matches],
            }
        )

    if reference_fn is not None:
        xs = np.linspace(proxy_grid[0], proxy_grid[1], 301)
        ys = np.asarray(reference_fn(xs), dtype=float)
        agree = 0
        for result in results:
            pred = np.asarray(_evaluate(result.best_of_run, xs), dtype=float)
            if np.all(np.isfinite(pred)) and np.all(np.abs(pred - ys) <= proxy_tolerance):
                agree += 1
        summary["functional_agreement_proxy_rate"] = agree / n
        summary["functional_agreement_proxy_note"] = (
            f"在 [{proxy_grid[0]}, {proxy_grid[1]}] 的 301 点网格上逐点误差 ≤ "
            f"{proxy_tolerance:g} 的运行比例；只是数值代理，不是结构恢复证明。"
        )
    return summary


def _evaluate(node, xs):
    """延迟导入 evaluate，避免 experiments 在模块导入期携带无关依赖。"""
    from .operators import evaluate

    return evaluate(node, xs)


# ---------------------------------------------------------------------------
# 2) 数据生成：噪声、划分
# ---------------------------------------------------------------------------
def add_gaussian_noise(y, noise_ratio: float, rng):
    """给标签加高斯噪声：``sigma = noise_ratio * std(y)``（``ddof=0``）。

    参数
    ----
    y : array-like
        干净标签。
    noise_ratio : float
        噪声尺度系数，必须 ≥ 0；``0`` 时返回原值副本且 ``sigma = 0``。
    rng : numpy.random.Generator | random.Random
        优先传 ``numpy.random.default_rng(seed)``（可复现）；也兼容
        ``random.Random``（逐点调用 ``gauss``）。

    返回
    ----
    ``(y_noisy, sigma)``。噪声样本独立同分布 ``N(0, sigma^2)``。
    """
    values = np.asarray(y, dtype=float)
    if noise_ratio < 0:
        raise ValueError("noise_ratio 必须 ≥ 0")
    sigma = float(noise_ratio) * float(np.std(values, ddof=0))
    if sigma == 0.0:
        return values.copy(), 0.0
    if hasattr(rng, "normal"):
        noise = np.asarray(rng.normal(0.0, sigma, size=values.shape), dtype=float)
    else:  # random.Random 兼容路径
        noise = np.asarray([rng.gauss(0.0, sigma) for _ in values], dtype=float)
    return values + noise, sigma


def make_train_val_test(
    target_fn=koza1_target,
    *,
    n_train: int = 20,
    n_val: int = 10,
    n_test: int = 10,
    train_range: tuple[float, float] = (-1.0, 0.9),
    val_range: tuple[float, float] = (1.0, 1.4),
    test_range: tuple[float, float] = (1.5, 2.0),
    noise_ratio: float = 0.0,
    seed: int | None = None,
) -> dict:
    """构造互不重叠的训练/验证/测试集，并可选地给**训练标签**加高斯噪声。

    - 三个区间默认互不相交：``[-1, 0.9]``、``[1.0, 1.4]``、``[1.5, 2.0]``；
      函数内部会用 ``np.intersect1d`` 断言三组 x 没有公共点；
    - 验证集与测试集标签**不加噪声**（干净标签），训练集可选加噪；
    - 数据随机性（``seed``）与 GP 搜索随机性（``GPConfig.seed``）分离，
      因此噪声可复现、且与搜索种子无关。

    返回 dict，含 ``x_train`` / ``y_train``（加噪后的训练标签）/ ``y_train_clean``
    / ``y_train_noisy`` / ``x_val`` / ``y_val`` / ``x_test`` / ``y_test``
    / ``sigma`` / ``meta``。``x_train, y_train, x_test, y_test`` 四个键可直接传给
    ``EvolutionEngine(data=split)``。
    """
    if noise_ratio < 0:
        raise ValueError("noise_ratio 必须 ≥ 0")
    x_train = np.linspace(train_range[0], train_range[1], n_train)
    x_val = np.linspace(val_range[0], val_range[1], n_val)
    x_test = np.linspace(test_range[0], test_range[1], n_test)
    for name_a, a, name_b, b in (
        ("train", x_train, "val", x_val),
        ("train", x_train, "test", x_test),
        ("val", x_val, "test", x_test),
    ):
        if np.intersect1d(a, b).size != 0:
            raise AssertionError(f"{name_a} 与 {name_b} 的 x 有重叠，划分不独立")

    y_train_clean = np.asarray(target_fn(x_train), dtype=float)
    y_val = np.asarray(target_fn(x_val), dtype=float)
    y_test = np.asarray(target_fn(x_test), dtype=float)

    if noise_ratio > 0:
        noise_rng = np.random.default_rng(seed)
        y_train_noisy, sigma = add_gaussian_noise(y_train_clean, noise_ratio, noise_rng)
    else:
        y_train_noisy, sigma = y_train_clean.copy(), 0.0

    return {
        "x_train": x_train,
        "y_train": y_train_noisy,
        "y_train_clean": y_train_clean,
        "y_train_noisy": y_train_noisy,
        "x_val": x_val,
        "y_val": y_val,
        "x_test": x_test,
        "y_test": y_test,
        "sigma": sigma,
        "meta": {
            "n_train": n_train,
            "n_val": n_val,
            "n_test": n_test,
            "train_range": tuple(train_range),
            "val_range": tuple(val_range),
            "test_range": tuple(test_range),
            "noise_ratio": float(noise_ratio),
            "noise_sigma": sigma,
            "noise_ddof": 0,
            "data_seed": seed,
            "target": getattr(target_fn, "__name__", repr(target_fn)),
        },
    }


# ---------------------------------------------------------------------------
# 3) 控制变量实验（初始化 / 选择 / 预算 / 简约压力）
# ---------------------------------------------------------------------------
def initialization_experiment(
    config: GPConfig,
    methods=INIT_METHODS,
    *,
    depth: int | None = None,
    population_size: int | None = None,
    seed: int = 0,
) -> list[dict]:
    """对每个初始化方法各生成一个种群，返回深度/规模/多样性统计。

    每个方法都从**相同 RNG 初态**（``make_rng(seed)``）开始，保证可比。
    ``depth`` 默认取 ``config.max_init_depth``；``population_size`` 默认取配置值。
    """
    cfg = replace(
        config,
        max_init_depth=depth if depth is not None else config.max_init_depth,
        population_size=population_size if population_size is not None else config.population_size,
    )
    rows: list[dict] = []
    for method in methods:
        rng = make_rng(seed)
        population = initial_population(cfg, rng, method=method)
        stats = population_stats(population, None)
        rows.append(
            {
                "method": method,
                "depth_setting": cfg.max_init_depth,
                **stats,
            }
        )
    return rows


def tournament_experiment(
    config: GPConfig,
    population,
    fitnesses,
    tournament_sizes=(1, 2, 7, 12),
    *,
    n_select: int = 40,
    seed: int = 0,
) -> list[dict]:
    """固定种群与适应度，只改变锦标赛规模 k，统计父本适应度与结构多样性。

    每个 k 都从相同 RNG 初态开始（``make_rng(seed)``），因此差异只来自 k。
    返回每个 k 一行：父本适应度分布、被选中父本的结构唯一数、以及选中下标，
    便于复算。注意：本函数只观察**选择一步**的结果，不代表完整进化的最终优劣。
    """
    rows: list[dict] = []
    for k in tournament_sizes:
        cfg = replace(config, tournament_size=int(k))
        indices = tournament_select_indices(population, fitnesses, n_select, cfg, make_rng(seed))
        parent_fitnesses = [float(fitnesses[i]) for i in indices]
        signatures = [structural_signature(population[i]) for i in indices]
        rows.append(
            {
                "tournament_size": int(k),
                "n_selected": int(n_select),
                "parent_fitnesses": parent_fitnesses,
                "parent_fitness_mean": float(np.mean(parent_fitnesses)),
                "parent_fitness_median": float(np.median(parent_fitnesses)),
                "parent_fitness_min": float(np.min(parent_fitnesses)),
                "parent_fitness_max": float(np.max(parent_fitnesses)),
                "population_fitness_mean": float(np.mean(fitnesses)),
                "unique_parent_structures": len(set(signatures)),
                "unique_parent_ratio": len(set(signatures)) / len(signatures),
                "selected_indices": [int(i) for i in indices],
            }
        )
    return rows


def run_budget_experiment(
    config: GPConfig, seeds, budgets, *, verbose: bool = False
) -> dict[int, list[RunResult]]:
    """同一组种子、不同 ``max_generations`` 预算，返回 ``{budget: [RunResult, ...]}``。

    除了预算外不改变任何配置（控制变量）；所有种子结果全部保留。
    """
    out: dict[int, list[RunResult]] = {}
    for budget in budgets:
        cfg = replace(config, max_generations=int(budget))
        if verbose:
            print(f"--- budget = {budget} generations ---")
        out[int(budget)] = run_seed_batch(cfg, seeds, verbose=verbose)
    return out


def make_parsimony_fitness(penalty: float):
    """构造简约压力目标：``fitness'(tree) = MSE(tree) + penalty * size(tree)``。

    这是**实验变体**，不是原 Koza-1 算法：默认 ``penalty=0`` 时与原目标完全一致，
    ``penalty>0`` 时选择会同时偏好小树。注意此时 ``best_fitness`` 与 MSE 量纲不同，
    快照里的 ``best_train_mse`` 仍是原始 MSE，因此可以单独观察「误差-复杂度」权衡。
    """
    if penalty < 0:
        raise ValueError("penalty 必须 ≥ 0")

    def parsimony_fitness(node, x, y) -> float:
        return fitness(node, x, y) + penalty * node.size()

    parsimony_fitness.penalty = penalty  # type: ignore[attr-defined]
    return parsimony_fitness


# ---------------------------------------------------------------------------
# 4) 结果保存与读取
# ---------------------------------------------------------------------------
def _jsonable(value):
    """把 numpy / dataclass / Node / tuple 等递归转成 JSON 可序列化对象。"""
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if is_dataclass(value) and not isinstance(value, type):
        return _jsonable(asdict(value))
    if hasattr(value, "sexpr_str"):
        return value.sexpr_str()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def result_to_dict(result: RunResult) -> dict:
    """把 ``RunResult`` 压成可独立阅读/保存的字典（不含完整种群）。"""
    return {
        "seed": result.seed,
        "best_expression": result.best_of_run.sexpr_str(),
        "best_infix": result.best_of_run.infix_str(),
        "train_mse": float(result.train_mse),
        "test_mse": float(result.test_mse),
        "gen0_mse": float(result.gen0_mse),
        "generation_found": result.generation_found,
        "n_generations": int(result.n_generations),
        "evaluations": int(result.evaluations),
        "stop_reason": result.stop_reason,
        "best_mse_history": [float(v) for v in result.best_mse_history],
        "test_mse_history": [float(v) for v in result.test_mse_history],
        "snapshots": [
            {
                "generation": s.generation,
                "best_expression": s.best_expression,
                "best_train_mse": float(s.best_train_mse),
                "best_test_mse": float(s.best_test_mse),
                "best_fitness": float(s.best_fitness),
                "mean_fitness": float(s.mean_fitness),
                "median_fitness": float(s.median_fitness),
                "worst_fitness": float(s.worst_fitness),
                "mean_size": float(s.mean_size),
                "max_size": int(s.max_size),
                "mean_depth": float(s.mean_depth),
                "max_depth": int(s.max_depth),
                "unique_structures": int(s.unique_structures),
                "unique_ratio": float(s.unique_ratio),
                "evaluations": int(s.evaluations),
            }
            for s in result.snapshots
        ],
    }


def save_experiment(path, config: GPConfig, results, summary: dict) -> Path:
    """把配置、种子、全部运行结果与摘要写成可独立读取的 JSON 文件。

    返回实际写入的 ``Path``；文件内容含 ``format_version``、``saved_at``、
    ``config``、``seeds``、``results``、``summary``。该函数不做任何筛选。
    """
    output = Path(path)
    payload = {
        "format_version": 1,
        "saved_at": datetime.now().isoformat(timespec="seconds"),
        "config": _jsonable(config),
        "seeds": [r.seed for r in results],
        "results": [result_to_dict(r) for r in results],
        "summary": _jsonable(summary),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return output


def load_experiment(path) -> dict:
    """读取 ``save_experiment`` 写出的 JSON，返回原始 dict。"""
    return json.loads(Path(path).read_text(encoding="utf-8"))
