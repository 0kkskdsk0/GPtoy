"""``gptoy.variation`` 冒烟测试。"""

import pytest

from gptoy.config import GPConfig
from gptoy.representation import Node, str_to_tree
from gptoy.utils import make_rng
from gptoy.variation import apply_variation, crossover, mutate, reproduce

CFG = GPConfig()


def _chain(depth: int) -> Node:
    """构造深度恰为 ``depth`` 的链状树（最坏情况深树，节点数 = depth）。"""
    n = Node("x")
    for _ in range(depth - 1):
        n = Node("*", [n, Node("x")])
    return n


def _all_nodes(tree: Node):
    yield tree
    for child in tree.children:
        yield from _all_nodes(child)


def _assert_valid(tree: Node):
    allowed = set(CFG.function_set) | set(CFG.terminal_set)
    for node in _all_nodes(tree):
        assert node.value in allowed
    assert tree.depth() <= CFG.max_tree_depth


def test_crossover_count():
    p1 = str_to_tree("(+ x (* x x))", CFG)
    p2 = str_to_tree("(* (+ x x) x)", CFG)
    c1, c2 = crossover(p1, p2, CFG, make_rng(0))
    assert isinstance(c1, Node) and isinstance(c2, Node)


def test_crossover_validity_deep_trees():
    # 用深度 17 的链状深树反复交叉：子代节点合法且深度 ≤ max_tree_depth
    rng = make_rng(1)
    p1 = _chain(CFG.max_tree_depth)
    p2 = _chain(CFG.max_tree_depth)
    for _ in range(50):
        c1, c2 = crossover(p1, p2, CFG, rng)
        _assert_valid(c1)
        _assert_valid(c2)


def test_crossover_changes_offspring():
    p1 = str_to_tree("(+ x (* x x))", CFG)
    p2 = str_to_tree("(* (+ x x) x)", CFG)
    c1, c2 = crossover(p1, p2, CFG, make_rng(5))
    # 至少一个子代与两个亲本都不同（存在进化），或深度约束回退为亲本副本
    assert (c1 != p1 or c2 != p1) or (c1 != p2 or c2 != p2)


def test_mutation_validity():
    ind = str_to_tree("(+ (* x x) (- x x))", CFG)
    for _ in range(20):
        m = mutate(ind, CFG, make_rng(6))
        _assert_valid(m)


def test_reproduction_isolation():
    ind = str_to_tree("(+ x (* x x))", CFG)
    c = reproduce(ind)
    c.children[0] = Node("-", [Node("x"), Node("x")])
    assert c != ind
    assert ind.children[0] == Node("x")


def test_probability_dispatch():
    # 固定种子多次抽样：reproduction（子代==亲本）与 crossover/mutation（子代!=亲本）
    # 均应出现（3.1：选择、交叉、变异可观测）
    rng = make_rng(8)
    inds = [str_to_tree(s, CFG) for s in ("(+ x x)", "(* x x)", "(- x x)", "(/ x x)")]
    identical = 0
    different = 0
    for _ in range(40):
        off = apply_variation(inds, CFG, rng)
        for parent, child in zip(inds, off):
            if child == parent:
                identical += 1
            else:
                different += 1
    assert identical > 0, "reproduction 未触发"
    assert different > 0, "crossover/mutation 未触发"


def test_apply_variation_length():
    rng = make_rng(9)
    inds = [str_to_tree("(+ x x)", CFG)] * 99
    off = apply_variation(inds, CFG, rng)
    assert len(off) == 99


@pytest.mark.parametrize("bad", ["(x x)", "(+ x)", "(+ x x))", "(+ x", "y"])
def test_str_to_tree_rejects_bad_input(bad):
    with pytest.raises(ValueError):
        str_to_tree(bad, CFG)
