"""``gptoy.selection`` 测试：数量、压力、有放回、确定性、下标版本一致性。"""

import pytest

from gptoy.config import GPConfig
from gptoy.representation import Node
from gptoy.selection import tournament_select, tournament_select_indices
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


def test_stronger_pressure_picks_smaller_mse():
    # 构造一个适应度梯度明显的种群；k 越大，选中父本平均 MSE 应更接近最优
    pop = [A, B, A, B, A]
    fits = [0.0, 1.0, 2.0, 3.0, 4.0]
    idx1 = tournament_select_indices(pop, fits, 200, GPConfig(tournament_size=1), make_rng(11))
    idx5 = tournament_select_indices(pop, fits, 200, GPConfig(tournament_size=5), make_rng(11))
    mean1 = sum(fits[i] for i in idx1) / 200
    mean5 = sum(fits[i] for i in idx5) / 200
    assert mean5 < mean1


def test_deterministic_with_seed():
    pop = [A, B, A, B, A]
    fits = [0.1, 0.2, 0.1, 0.2, 0.5]
    cfg = GPConfig()
    r1 = [p.sexpr_str() for p in tournament_select(pop, fits, 20, cfg, make_rng(7))]
    r2 = [p.sexpr_str() for p in tournament_select(pop, fits, 20, cfg, make_rng(7))]
    assert r1 == r2


def test_indices_match_select():
    """下标版本与节点版本必须消费相同随机数、给出一致结果。"""
    pop = [A, B, A, B, A]
    fits = [0.1, 0.2, 0.1, 0.2, 0.5]
    cfg = GPConfig()
    indices = tournament_select_indices(pop, fits, 20, cfg, make_rng(7))
    nodes = tournament_select(pop, fits, 20, cfg, make_rng(7))
    assert [pop[i].sexpr_str() for i in indices] == [n.sexpr_str() for n in nodes]
    assert all(0 <= i < len(pop) for i in indices)


def test_mismatched_lengths_raise():
    with pytest.raises(ValueError):
        tournament_select([A, B], [0.1], 2, GPConfig(), make_rng(0))
    with pytest.raises(ValueError):
        tournament_select_indices([A, B], [0.1], 2, GPConfig(), make_rng(0))


def test_negative_k_raises():
    with pytest.raises(ValueError):
        tournament_select([A], [0.0], -1, GPConfig(), make_rng(0))


def test_empty_population_raises():
    with pytest.raises(ValueError):
        tournament_select_indices([], [], 1, GPConfig(), make_rng(0))
