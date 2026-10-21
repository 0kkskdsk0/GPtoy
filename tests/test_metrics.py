"""``gptoy.metrics`` 测试：结构签名一致性与可手算的种群统计。"""

import math

import pytest

from gptoy.config import GPConfig
from gptoy.metrics import complexity, is_finite_fitness, population_stats, structural_signature
from gptoy.representation import Node, str_to_tree

CFG = GPConfig()


def test_structural_signature_consistent_with_eq():
    a = str_to_tree("(+ x (* x x))", CFG)
    b = str_to_tree("(+ x (* x x))", CFG)
    c = str_to_tree("(+ (* x x) x)", CFG)
    assert structural_signature(a) == structural_signature(b) == a.sexpr_str()
    assert (a == b) is (structural_signature(a) == structural_signature(b))
    assert structural_signature(a) != structural_signature(c)


def test_population_stats_hand_computed():
    pop = [Node("x"), str_to_tree("(+ x x)", CFG), str_to_tree("(+ x (* x x))", CFG)]
    fits = [4.0, 2.0, 0.0]
    stats = population_stats(pop, fits, generation=7)

    assert stats["generation"] == 7
    assert stats["population_size"] == 3
    assert stats["best_fitness"] == 0.0
    assert stats["worst_fitness"] == 4.0
    assert stats["mean_fitness"] == pytest.approx(2.0)
    assert stats["median_fitness"] == pytest.approx(2.0)
    assert stats["fitness_std"] == pytest.approx(math.sqrt(8 / 3))  # ddof=0

    assert stats["mean_size"] == pytest.approx(3.0)  # (1 + 3 + 5) / 3
    assert stats["max_size"] == 5
    assert stats["mean_depth"] == pytest.approx(2.0)  # (1 + 2 + 3) / 3
    assert stats["max_depth"] == 3
    assert stats["unique_structures"] == 3
    assert stats["unique_ratio"] == pytest.approx(1.0)
    assert stats["size_histogram"] == {1: 1, 3: 1, 5: 1}
    assert stats["depth_histogram"] == {1: 1, 2: 1, 3: 1}


def test_population_stats_duplicates_and_inf():
    t = str_to_tree("(+ x x)", CFG)
    pop = [t.deepcopy(), t.deepcopy(), Node("x")]
    stats = population_stats(pop, [1.0, 1.0, float("inf")])
    assert stats["unique_structures"] == 2
    assert stats["unique_ratio"] == pytest.approx(2 / 3)
    assert stats["worst_fitness"] == float("inf")
    assert stats["best_fitness"] == 1.0


def test_population_stats_without_fitnesses():
    stats = population_stats([Node("x")])
    assert stats["best_fitness"] is None
    assert stats["mean_fitness"] is None
    assert stats["mean_size"] == 1.0


def test_population_stats_validation():
    with pytest.raises(ValueError):
        population_stats([])
    with pytest.raises(ValueError):
        population_stats([Node("x")], [1.0, 2.0])


def test_complexity():
    t = str_to_tree("(+ x (* x x))", CFG)
    c = complexity(t)
    assert c["size"] == 5 and c["depth"] == 3
    assert c["node_counts"] == {"+": 1, "*": 1, "x": 3}
    assert c["sexpr"] == t.sexpr_str()


def test_is_finite_fitness():
    assert is_finite_fitness(0.0)
    assert not is_finite_fitness(float("inf"))
    assert not is_finite_fitness(float("nan"))
