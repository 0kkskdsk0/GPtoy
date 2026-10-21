"""``gptoy.evolution`` 测试：Koza-1 基线、逐代/批量一致、tracer 不变性、评估计数。

所有测试都使用**写死的常量种子**（``SEED = 3``），不依赖 ``results/found_seed.json``
等未纳入版本控制的前置文件。
"""

from dataclasses import replace

import numpy as np
import pytest

from gptoy.config import GPConfig
from gptoy.data import make_koza1_data
from gptoy.evolution import EvolutionEngine, RunResult, run_gp
from gptoy.operators import fitness
from gptoy.tracing import EvolutionTracer, NullTracer

SEED = 3

# 小型配置：用于「逐代 vs 批量 / tracer 开关」这类需要多次运行的等价性测试。
SMALL = GPConfig(
    seed=0,
    population_size=20,
    max_generations=6,
    max_tree_depth=7,
    min_init_depth=2,
    max_init_depth=4,
)


@pytest.fixture(scope="module")
def baseline() -> RunResult:
    """默认配置、seed=3 的 Koza-1 基线（只跑一次，供多个断言复用）。"""
    return run_gp(GPConfig(seed=SEED))


def test_koza1_baseline_converges(baseline):
    assert baseline.train_mse < 1e-6, f"训练 MSE {baseline.train_mse} 未达标"
    assert baseline.test_mse < 1e-6, f"泛化 MSE {baseline.test_mse} 未达标"
    assert baseline.generation_found is not None
    assert baseline.generation_found <= 50
    assert baseline.n_generations <= 50


def test_gen0_not_solved(baseline):
    assert baseline.gen0_mse > 1e-3, f"第 0 代即解出（gen0_mse={baseline.gen0_mse}）"


def test_reproducible(baseline):
    again = run_gp(GPConfig(seed=SEED))
    assert again.best_mse_history == baseline.best_mse_history
    assert again.best_of_run.sexpr_str() == baseline.best_of_run.sexpr_str()
    assert again.train_mse == baseline.train_mse
    assert again.test_mse == baseline.test_mse
    assert again.evaluations == baseline.evaluations


def test_budget_and_history(baseline):
    assert baseline.n_generations == len(baseline.best_mse_history) - 1
    assert len(baseline.snapshots) == len(baseline.best_mse_history)
    assert baseline.train_mse_history == baseline.best_mse_history
    assert len(baseline.test_mse_history) == len(baseline.snapshots)


def test_evaluations_match_population_evaluations(baseline):
    """评估次数 = 种群规模 × 评价过的代数（不含测试评价与诊断评价）。"""
    cfg = GPConfig(seed=SEED)
    assert baseline.evaluations == cfg.population_size * len(baseline.snapshots)
    assert [s.evaluations for s in baseline.snapshots] == [
        cfg.population_size * (i + 1) for i in range(len(baseline.snapshots))
    ]


def test_snapshot_fields(baseline):
    s = baseline.snapshots[-1]
    assert s.generation == baseline.n_generations
    assert s.best_expression == baseline.best_of_run.sexpr_str()
    assert s.best_train_mse == pytest.approx(baseline.train_mse)
    assert s.best_test_mse == pytest.approx(baseline.test_mse)
    assert 1 <= s.unique_structures <= GPConfig().population_size
    assert 0 < s.unique_ratio <= 1.0
    assert sum(s.size_histogram.values()) == GPConfig().population_size
    assert sum(s.depth_histogram.values()) == GPConfig().population_size
    assert s.mean_size <= s.max_size
    assert s.mean_depth <= s.max_depth
    assert s.best_fitness <= s.mean_fitness <= s.worst_fitness


def test_hall_of_fame(baseline):
    assert len(baseline.hall_of_fame) >= 1
    assert baseline.hall_of_fame[-1].sexpr_str() == baseline.best_of_run.sexpr_str()


def test_stop_reason(baseline):
    assert baseline.stop_reason in {"converged", "max_generations"}
    assert baseline.stop_reason == "converged"


def test_step_equals_run():
    """逐代 step() 与 run_gp() 必须共享同一引擎、产生一致历史。"""
    r1 = run_gp(SMALL)
    engine = EvolutionEngine(SMALL)
    snapshot0 = engine.initialize()
    assert snapshot0.generation == 0
    assert engine.generation == 0 and engine.done is False
    while not engine.done:
        snapshot = engine.step()
        assert snapshot is not None and snapshot.generation == engine.generation
    r2 = engine.result

    assert r1.best_mse_history == r2.best_mse_history
    assert r1.best_of_run.sexpr_str() == r2.best_of_run.sexpr_str()
    assert r1.evaluations == r2.evaluations
    assert r1.n_generations == r2.n_generations
    assert r1.train_mse == r2.train_mse and r1.test_mse == r2.test_mse
    assert [s.best_expression for s in r1.snapshots] == [s.best_expression for s in r2.snapshots]


def test_step_after_done_returns_none():
    engine = EvolutionEngine(SMALL)
    engine.initialize()
    while not engine.done:
        engine.step()
    calls = engine.evaluations
    assert engine.step() is None
    assert engine.evaluations == calls


def test_tracer_does_not_change_trajectory():
    r_plain = run_gp(SMALL)
    r_traced = run_gp(SMALL, tracer=EvolutionTracer())
    assert r_traced.best_mse_history == r_plain.best_mse_history
    assert r_traced.best_of_run.sexpr_str() == r_plain.best_of_run.sexpr_str()
    assert r_traced.evaluations == r_plain.evaluations


def test_null_tracer_equals_no_tracer():
    r_plain = run_gp(SMALL)
    r_null = run_gp(SMALL, tracer=NullTracer())
    assert r_null.best_mse_history == r_plain.best_mse_history
    assert r_null.best_of_run.sexpr_str() == r_plain.best_of_run.sexpr_str()


def test_stop_on_target_can_be_disabled():
    """关闭提前停止后必须跑满 max_generations，但仍记录首达代。"""
    cfg = replace(
        GPConfig(seed=0),
        population_size=20,
        max_generations=8,
        max_tree_depth=7,
        stop_on_target=False,
    )
    result = run_gp(cfg)
    assert result.n_generations == 8
    assert result.generation_found is not None and result.generation_found < 8
    assert result.stop_reason == "max_generations"
    assert result.evaluations == 20 * 9  # 第 0 代 + 8 代


def test_custom_data_is_used_for_test_evaluation():
    x_train, y_train, x_test, y_test = make_koza1_data(GPConfig())
    custom = (x_train, y_train, x_test, y_test * 2.0)
    result = run_gp(
        GPConfig(seed=1, population_size=8, max_generations=1, max_tree_depth=5),
        data=custom,
    )
    expected = fitness(result.best_of_run, x_test, y_test * 2.0)
    assert result.test_mse == pytest.approx(expected)


def test_custom_fitness_keeps_raw_mse_in_snapshot():
    """自定义搜索目标（简约惩罚）只影响选择，快照仍记录原始训练 MSE。"""
    from gptoy.experiments import make_parsimony_fitness

    cfg = GPConfig(seed=1, population_size=10, max_generations=2, max_tree_depth=5)
    result = run_gp(cfg, fitness_fn=make_parsimony_fitness(0.05))
    s = result.snapshots[0]
    assert s.best_fitness > s.best_train_mse  # 惩罚项 > 0
    x_train, y_train, _, _ = make_koza1_data(cfg)
    assert result.train_mse == pytest.approx(
        fitness(result.best_of_run, x_train, y_train)
    )


def test_run_result_config_and_seed():
    result = run_gp(SMALL)
    assert result.config is SMALL
    assert result.seed == SMALL.seed
    assert isinstance(result, RunResult)


def test_invalid_configs_raise():
    with pytest.raises(ValueError):
        EvolutionEngine(GPConfig(population_size=10, elitism_size=11))
    with pytest.raises(ValueError):
        EvolutionEngine(GPConfig(p_crossover=0.8, p_mutation=0.5))
    with pytest.raises(ValueError):
        EvolutionEngine(GPConfig(tournament_size=0))
    with pytest.raises(ValueError):
        EvolutionEngine(GPConfig(min_init_depth=5, max_init_depth=3))


def test_all_infinite_fitness_does_not_crash():
    """所有个体 fitness=inf 时也必须能构造 RunResult（best_of_run 不能为 None）。"""
    inf_fitness = lambda node, x, y: float("inf")  # noqa: E731

    cfg = GPConfig(seed=0, population_size=5, max_generations=0)
    result = run_gp(cfg, fitness_fn=inf_fitness)
    assert result.best_of_run is not None
    # 搜索目标全为 inf：train_mse 报告 inf（没有有限目标值），
    # 但测试集仍用原始 MSE 评价，因此 test_mse 是有限的。
    assert result.train_mse == float("inf")
    assert np.isfinite(result.test_mse)
    assert len(result.hall_of_fame) == 1
    assert result.generation_found is None
    assert result.stop_reason == "max_generations"

    # 逐代路径同样不能崩：initialize 后即使 best_fitness 为 inf 也要有 best_of_run
    step_cfg = GPConfig(seed=0, population_size=5, max_generations=1)
    engine = EvolutionEngine(step_cfg, fitness_fn=inf_fitness)
    engine.initialize()
    assert engine.best_of_run is not None
    assert engine.result.best_of_run is not None
    while not engine.done:
        engine.step()
    stepped = engine.result
    assert stepped.best_of_run is not None
    assert stepped.train_mse == float("inf")
    assert np.isfinite(stepped.test_mse)
    assert stepped.n_generations == 1


def test_unstarted_engine_raises():
    engine = EvolutionEngine(SMALL)
    with pytest.raises(RuntimeError):
        engine.step()
    with pytest.raises(RuntimeError):
        _ = engine.result
