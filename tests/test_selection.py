"""``gptoy.selection`` 冒烟测试。"""

from gptoy.config import GPConfig
from gptoy.representation import Node
from gptoy.selection import tournament_select
from gptoy.utils import make_rng

A = Node("x")
B = Node("+", [Node("x"), Node("x")])


def test_select_count():
    pop = [A, B, A, B]
    fits = [0.1, 0.2, 0.3, 0.4]
    parents = tournament_select(pop, fits, k=10, config=GPConfig(), rng=make_rng(0))
    assert len(parents) == 10


def test_selection_pressure():
    # 种群只有 2 个个体：MSE 0 与 MSE 1；锦标赛大小 2（有放回抽 2 个）
    # P(最优者胜出) = 1 - (1/2)^2 = 0.75，100 次抽样期望 75 次
    cfg = GPConfig(tournament_size=2)
    parents = tournament_select([A, B], [0.0, 1.0], k=100, config=cfg, rng=make_rng(1))
    n_best = sum(1 for p in parents if p == A)
    assert n_best >= 60


def test_with_replacement():
    # 锦标赛大小 1 时每轮等概率抽 1 个：两个个体都应出现（有放回抽样）
    cfg = GPConfig(tournament_size=1)
    parents = tournament_select([A, B], [0.0, 0.0], k=50, config=cfg, rng=make_rng(2))
    assert len(parents) == 50
    assert {p.value for p in parents} == {"x", "+"}
    # 极端情形：单个体种群允许同一亲本被反复选中
    only = tournament_select([A], [0.0], k=5, config=cfg, rng=make_rng(3))
    assert only == [A, A, A, A, A]


def test_deterministic_with_seed():
    pop = [A, B, A, B, A]
    fits = [0.1, 0.2, 0.1, 0.2, 0.5]
    cfg = GPConfig()
    r1 = [p.sexpr_str() for p in tournament_select(pop, fits, 20, cfg, make_rng(7))]
    r2 = [p.sexpr_str() for p in tournament_select(pop, fits, 20, cfg, make_rng(7))]
    assert r1 == r2


def test_mismatched_lengths_raise():
    import pytest

    with pytest.raises(ValueError):
        tournament_select([A, B], [0.1], 2, GPConfig(), make_rng(0))
