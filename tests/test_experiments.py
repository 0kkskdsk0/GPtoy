"""``gptoy.experiments`` 测试：划分/噪声、多种子保留、指标分离、保存读取、控制变量。"""

import json
from dataclasses import replace

import numpy as np
import pytest

from gptoy.config import GPConfig
from gptoy.evolution import EvolutionEngine, run_gp
from gptoy.experiments import (
    add_gaussian_noise,
    initialization_experiment,
    load_experiment,
    make_parsimony_fitness,
    make_train_val_test,
    result_to_dict,
    run_budget_experiment,
    run_seed_batch,
    save_experiment,
    summarize_runs,
    tournament_experiment,
)
from gptoy.operators import fitness
from gptoy.representation import str_to_tree
from gptoy.utils import make_rng

BASE = GPConfig(seed=0)
SMALL = replace(
    BASE,
    population_size=20,
    max_generations=6,
    max_tree_depth=7,
)
SEEDS = (0, 1, 2)


# ---------------------------------------------------------------------------
# 数据划分与噪声
# ---------------------------------------------------------------------------
def test_make_train_val_test_disjoint_and_sized():
    split = make_train_val_test(seed=1)
    assert split["x_train"].shape == (20,)
    assert split["x_val"].shape == (10,)
    assert split["x_test"].shape == (10,)
    for a, b in (
        ("x_train", "x_val"),
        ("x_train", "x_test"),
        ("x_val", "x_test"),
    ):
        assert np.intersect1d(split[a], split[b]).size == 0, f"{a} 与 {b} 重叠"
    assert np.allclose(split["y_test"], split["x_test"] ** 4 + split["x_test"] ** 3 + split["x_test"] ** 2 + split["x_test"])
    assert split["sigma"] == 0.0
    assert np.array_equal(split["y_train"], split["y_train_clean"])


def test_noise_formula_and_reproducibility():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    noisy_a, sigma_a = add_gaussian_noise(y, 0.1, np.random.default_rng(0))
    noisy_b, sigma_b = add_gaussian_noise(y, 0.1, np.random.default_rng(0))
    noisy_c, sigma_c = add_gaussian_noise(y, 0.1, np.random.default_rng(1))
    assert sigma_a == sigma_b == sigma_c == pytest.approx(0.1 * np.std(y, ddof=0))
    assert np.allclose(noisy_a, noisy_b), "相同 data seed 必须得到相同噪声"
    assert not np.allclose(noisy_a, noisy_c), "不同 data seed 应得到不同噪声"
    # 噪声均值近似 0、量级合理
    assert abs(np.mean(noisy_a - y)) < 3 * sigma_a


def test_zero_noise_returns_copy():
    y = np.array([1.0, 2.0])
    out, sigma = add_gaussian_noise(y, 0.0, np.random.default_rng(0))
    assert sigma == 0.0 and np.array_equal(out, y)
    assert out is not y


def test_noise_ratio_must_be_non_negative():
    with pytest.raises(ValueError):
        add_gaussian_noise(np.array([1.0]), -0.1, np.random.default_rng(0))
    with pytest.raises(ValueError):
        make_train_val_test(noise_ratio=-0.1)


def test_split_noise_uses_data_seed_only():
    a = make_train_val_test(seed=5, noise_ratio=0.1)
    b = make_train_val_test(seed=5, noise_ratio=0.1)
    c = make_train_val_test(seed=6, noise_ratio=0.1)
    assert a["sigma"] == b["sigma"] == pytest.approx(0.1 * np.std(a["y_train_clean"], ddof=0))
    assert np.allclose(a["y_train_noisy"], b["y_train_noisy"])
    assert not np.allclose(a["y_train_noisy"], c["y_train_noisy"])
    # 验证/测试标签使用干净目标函数，不加噪声
    x_val = a["x_val"]
    assert np.allclose(a["y_val"], x_val**4 + x_val**3 + x_val**2 + x_val)
    assert not np.array_equal(a["y_train_noisy"], a["y_train_clean"])


def test_test_set_does_not_feed_back_into_search():
    """只改测试集标签，搜索轨迹与最优表达式必须完全不变。"""
    split = make_train_val_test(seed=2)
    data_a = (split["x_train"], split["y_train"], split["x_test"], split["y_test"])
    data_b = (split["x_train"], split["y_train"], split["x_test"], split["y_test"] * 3.0)
    r_a = run_gp(SMALL, data=data_a)
    r_b = run_gp(SMALL, data=data_b)
    assert r_a.best_mse_history == r_b.best_mse_history
    assert r_a.best_of_run.sexpr_str() == r_b.best_of_run.sexpr_str()
    assert r_a.evaluations == r_b.evaluations
    if np.isfinite(r_a.test_mse) and np.isfinite(r_b.test_mse):
        assert r_a.test_mse != pytest.approx(r_b.test_mse)


def test_validation_set_is_independent_and_usable():
    split = make_train_val_test(seed=3, n_train=10, n_val=5, n_test=5)
    # 验证集不参与引擎（引擎只接收 train/test），可用于独立选择检查
    result = run_gp(SMALL, data=split)
    val_mse = fitness(result.best_of_run, split["x_val"], split["y_val"])
    assert np.isfinite(val_mse)
    assert len(split["x_val"]) == 5


# ---------------------------------------------------------------------------
# 多种子批量与汇总
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def batch():
    return run_seed_batch(SMALL, SEEDS)


def test_run_seed_batch_keeps_all_seeds(batch):
    assert len(batch) == len(SEEDS)
    assert [r.seed for r in batch] == list(SEEDS)
    target = SMALL.mse_target
    # 成功与失败种子都被保留（不按测试集过滤）
    assert any(r.train_mse < target for r in batch)
    assert any(r.train_mse >= target for r in batch)


def test_summarize_keeps_metrics_separate(batch):
    summary = summarize_runs(batch)
    assert summary["n_runs"] == 3
    assert 0.0 <= summary["train_success_rate"] <= 1.0
    assert 0.0 <= summary["test_success_rate"] <= 1.0
    assert summary["train_success_count"] == sum(
        1 for r in batch if r.train_mse < SMALL.mse_target
    )
    assert summary["test_success_count"] == sum(
        1 for r in batch if r.test_mse < SMALL.mse_target
    )
    # 结构恢复默认不可用，必须显式为 None 而不是拿数值拟合冒充
    assert summary["structural_recovery_rate"] is None
    assert summary["structural_recovery_available"] is False
    assert "unavailable" in summary["structural_recovery_note"]
    assert summary["evaluations"]["total"] == sum(r.evaluations for r in batch)
    assert summary["first_hit_generation"]["count"] == sum(
        1 for r in batch if r.generation_found is not None
    )
    assert summary["failed_train_seeds"] == [
        r.seed for r in batch if r.train_mse >= SMALL.mse_target
    ]


def test_summarize_exact_structural_match_is_lower_bound(batch):
    # 故意给一个不存在的目标结构：结构恢复率应为 0，而训练可能已达标
    summary = summarize_runs(batch, strict_expression="(+ x x)")
    assert summary["structural_recovery_available"] is True
    assert summary["structural_recovery_rate"] == 0.0
    assert summary["train_success_rate"] > 0
    # 给出真实最优表达式时，匹配数至少 1
    expr = batch[0].best_of_run.sexpr_str()
    summary2 = summarize_runs(batch, strict_expression=expr)
    assert summary2["structural_recovery_count"] >= 1


def test_summarize_functional_proxy(batch):
    summary = summarize_runs(batch, reference_fn=lambda x: x**4 + x**3 + x**2 + x)
    assert 0.0 <= summary["functional_agreement_proxy_rate"] <= 1.0
    assert "数值代理" in summary["functional_agreement_proxy_note"]


def test_summarize_empty_raises():
    with pytest.raises(ValueError):
        summarize_runs([])


# ---------------------------------------------------------------------------
# 保存 / 读取
# ---------------------------------------------------------------------------
def test_save_and_load_experiment(tmp_path, batch):
    summary = summarize_runs(batch)
    path = tmp_path / "exp.json"
    written = save_experiment(path, SMALL, batch, summary)
    assert written.exists()
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["format_version"] == 1
    assert payload["config"]["population_size"] == SMALL.population_size
    assert payload["seeds"] == list(SEEDS)
    assert len(payload["results"]) == len(batch)
    assert payload["summary"]["n_runs"] == len(batch)
    assert set(payload["results"][0]) >= {
        "seed",
        "best_expression",
        "train_mse",
        "test_mse",
        "evaluations",
        "snapshots",
    }
    loaded = load_experiment(path)
    assert loaded["summary"]["train_success_rate"] == pytest.approx(
        summary["train_success_rate"]
    )


def test_result_to_dict_serializable(batch):
    d = result_to_dict(batch[0])
    json.dumps(d)  # 不抛异常即 JSON 可序列化
    assert d["best_expression"] == batch[0].best_of_run.sexpr_str()
    assert d["evaluations"] == batch[0].evaluations


# ---------------------------------------------------------------------------
# 预算 / 初始化 / 选择 / 简约压力
# ---------------------------------------------------------------------------
def test_budget_experiment_consumes_budget():
    cfg = replace(SMALL, stop_on_target=False)
    out = run_budget_experiment(cfg, [0, 1], (2, 4))
    assert set(out) == {2, 4}
    for budget, results in out.items():
        assert len(results) == 2
        for r in results:
            assert r.n_generations == budget
            # 关闭提前停止后，评估次数 = 种群 × (budget + 1)
            assert r.evaluations == cfg.population_size * (budget + 1)


def test_initialization_experiment():
    rows = initialization_experiment(
        BASE, ("full", "grow", "ramped"), depth=4, population_size=24, seed=0
    )
    assert [r["method"] for r in rows] == ["full", "grow", "ramped"]
    full_row, grow_row, ramped_row = rows
    assert full_row["max_depth"] == full_row["mean_depth"] == 4
    assert grow_row["max_depth"] <= 4
    assert len(ramped_row["depth_histogram"]) >= 2
    for row in rows:
        assert row["population_size"] == 24
        assert row["unique_structures"] >= 1
        assert sum(row["depth_histogram"].values()) == 24


def test_tournament_experiment_pressure():
    pop = [str_to_tree(s, BASE) for s in ("x", "(+ x x)", "(* x x)", "(- x x)", "(/ x x)")]
    fits = [4.0, 3.0, 2.0, 1.0, 0.0]
    rows = tournament_experiment(BASE, pop, fits, (1, 5), n_select=40, seed=0)
    assert [r["tournament_size"] for r in rows] == [1, 5]
    for row in rows:
        assert len(row["parent_fitnesses"]) == 40
        assert 0.0 < row["unique_parent_ratio"] <= 1.0
        assert row["parent_fitness_min"] >= min(fits)
        assert row["parent_fitness_max"] <= max(fits)
    # k 越大，父本平均适应度应更接近最优（选择压力更强）
    assert rows[1]["parent_fitness_mean"] < rows[0]["parent_fitness_mean"]


def test_parsimony_fitness_formula():
    tree = str_to_tree("(+ x x)", BASE)
    x = np.array([1.0, 2.0])
    y = x + x
    mse = fitness(tree, x, y)
    assert make_parsimony_fitness(0.0)(tree, x, y) == pytest.approx(mse)
    for penalty in (0.01, 0.5):
        expected = mse + penalty * tree.size()
        assert make_parsimony_fitness(penalty)(tree, x, y) == pytest.approx(expected)
    with pytest.raises(ValueError):
        make_parsimony_fitness(-1.0)


def test_engine_accepts_split_dict():
    split = make_train_val_test(seed=1)
    result = run_gp(SMALL, data=split)
    assert result.test_mse == pytest.approx(
        fitness(result.best_of_run, split["x_test"], split["y_test"])
    )
