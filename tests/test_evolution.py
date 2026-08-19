"""``gptoy.evolution`` 冒烟测试（对 Success Criteria 3.1 / 3.3 的直接验证）。"""

import json
from pathlib import Path

import pytest

from gptoy.config import GPConfig
from gptoy.evolution import run_gp

ROOT = Path(__file__).resolve().parents[1]
FOUND_SEED_PATH = ROOT / "results" / "found_seed.json"


def _found_seed() -> int | None:
    if not FOUND_SEED_PATH.exists():
        return None
    return int(json.loads(FOUND_SEED_PATH.read_text(encoding="utf-8"))["seed"])


@pytest.fixture(scope="module")
def found_seed():
    seed = _found_seed()
    if seed is None:
        pytest.skip("results/found_seed.json 不存在：请先运行 scripts/find_seed.py")
    return seed


def test_run_converges_with_found_seed(found_seed):
    r = run_gp(GPConfig(seed=found_seed))
    assert r.train_mse < 1e-6, f"训练 MSE {r.train_mse} 未达标（3.3）"
    assert r.test_mse < 1e-6, f"泛化 MSE {r.test_mse} 未达标（3.3）"
    assert r.generation_found is not None and r.generation_found <= 50


def test_gen0_not_solved(found_seed):
    r = run_gp(GPConfig(seed=found_seed))
    assert r.gen0_mse > 1e-3, f"第 0 代即解出（gen0_mse={r.gen0_mse}），进化不可观测（3.1）"


def test_reproducible(found_seed):
    r1 = run_gp(GPConfig(seed=found_seed))
    r2 = run_gp(GPConfig(seed=found_seed))
    assert r1.best_mse_history == r2.best_mse_history
    assert r1.best_of_run.sexpr_str() == r2.best_of_run.sexpr_str()
    assert r1.train_mse == r2.train_mse and r1.test_mse == r2.test_mse


def test_budget_respected(found_seed):
    r = run_gp(GPConfig(seed=found_seed))
    assert r.n_generations <= GPConfig().max_generations
    assert len(r.best_mse_history) == r.n_generations + 1
