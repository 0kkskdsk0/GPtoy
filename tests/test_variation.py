"""``gptoy.variation`` 测试：合法性、深度约束、回退、追踪与 RNG 不变性。"""

import pytest

from gptoy.config import GPConfig
from gptoy.representation import Node, str_to_tree
from gptoy.utils import make_rng
from gptoy.variation import (
    apply_variation,
    apply_variation_traced,
    crossover,
    crossover_traced,
    mutate,
    mutate_traced,
    reproduce,
)

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


def _assert_valid(tree: Node, cfg: GPConfig = CFG):
    allowed = set(cfg.function_set) | set(cfg.terminal_set)
    for node in _all_nodes(tree):
        assert node.value in allowed
    assert tree.depth() <= cfg.max_tree_depth


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
    """检查真实结构变化（而不是“与某个父本不同”这种在 p1!=p2 时恒真的弱条件）。

    交叉可能因为交换的子树结构相同而“数值/结构都不变”，也可能因深度回退而不变；
    两种情况都必须有明确原因，否则说明算子没有按交叉点交换子树。
    """
    p1 = str_to_tree("(+ x (* x x))", CFG)
    p2 = str_to_tree("(* (+ x x) x)", CFG)
    trace = crossover_traced(p1, p2, CFG, make_rng(5))
    changed = trace.child1 != p1 or trace.child2 != p2
    if not changed:
        # 结构不变只允许两种原因：深度回退，或交换的两棵子树结构完全相同
        assert trace.fallback or p1.subtree_at(trace.point1) == p2.subtree_at(trace.point2)
    else:
        # 确实变化时，两个交叉点必须是合法路径，且子代与普通接口一致
        c1, c2 = crossover(p1, p2, CFG, make_rng(5))
        assert trace.child1 == c1 and trace.child2 == c2
        assert trace.point1 in [path for _, path in p1.iter_nodes()]
        assert trace.point2 in [path for _, path in p2.iter_nodes()]


def test_crossover_fallback_recorded():
    """深树 + 很小的深度上限：重试耗尽后回退为父本副本，并如实记录 fallback。"""
    cfg = GPConfig(max_tree_depth=5)
    p1, p2 = _chain(17), _chain(17)
    trace = crossover_traced(p1, p2, cfg, make_rng(3))
    assert trace.fallback is True
    assert trace.child1 == p1 and trace.child2 == p2
    assert trace.child1 is not p1  # 回退也是深拷贝


def test_crossover_traced_points_are_valid_paths():
    p1 = str_to_tree("(+ x (* (- x x) x))", CFG)
    p2 = str_to_tree("(* (+ x x) (/ x x))", CFG)
    trace = crossover_traced(p1, p2, CFG, make_rng(9))
    paths1 = [path for _, path in p1.iter_nodes()]
    paths2 = [path for _, path in p2.iter_nodes()]
    assert trace.point1 in paths1
    assert trace.point2 in paths2
    # 交叉点处子树确实被交换
    expected1 = p1.replace_subtree(trace.point1, p2.subtree_at(trace.point2))
    assert trace.child1 == expected1


def test_traced_and_untraced_consume_same_rng():
    """记录追踪信息不得改变随机数消费顺序：同 seed 下子代逐位一致。"""
    p1 = str_to_tree("(+ x (* x x))", CFG)
    p2 = str_to_tree("(* (+ x x) (- x x))", CFG)
    c1, c2 = crossover(p1, p2, CFG, make_rng(13))
    trace = crossover_traced(p1, p2, CFG, make_rng(13))
    assert c1 == trace.child1 and c2 == trace.child2

    m = mutate(p1, CFG, make_rng(14))
    mt = mutate_traced(p1, CFG, make_rng(14))
    assert m == mt.child

    parents = [str_to_tree(s, CFG) for s in ("(+ x x)", "(* x x)", "(- x x)", "(/ x x)", "x")]
    plain = apply_variation(parents, CFG, make_rng(15))
    traced, records = apply_variation_traced(parents, CFG, make_rng(15))
    assert [t.sexpr_str() for t in plain] == [t.sexpr_str() for t in traced]
    assert len(records) > 0


def test_apply_variation_records_match_children():
    parents = [str_to_tree(s, CFG) for s in ("(+ x x)", "(* x x)", "(- x x)", "(/ x x)")]
    offspring, records = apply_variation_traced(parents, CFG, make_rng(16))
    # 记录里按顺序拼起来的子代必须正是输出列表
    flattened = [child for record in records for child in record.children]
    assert [c.sexpr_str() for c in flattened] == [c.sexpr_str() for c in offspring]
    assert {r.event_type for r in records} <= {"crossover", "mutation", "reproduction"}
    for record in records:
        assert all(0 <= p < len(parents) for p in record.parent_positions)
        if record.event_type == "crossover":
            assert len(record.children) == 2
            assert record.crossover_points is not None
            assert len(record.parent_positions) == 2
        else:
            assert len(record.children) == 1
        if record.event_type == "reproduction":
            assert record.changed is False
        if record.event_type == "mutation":
            assert record.mutation_point is not None


def test_mutation_validity():
    ind = str_to_tree("(+ (* x x) (- x x))", CFG)
    for _ in range(20):
        m = mutate(ind, CFG, make_rng(6))
        _assert_valid(m)


def test_mutation_point_and_deep_tree():
    ind = _chain(CFG.max_tree_depth)
    trace = mutate_traced(ind, CFG, make_rng(17))
    assert trace.point in [path for _, path in ind.iter_nodes()]
    assert trace.child.depth() <= CFG.max_tree_depth


def test_reproduction_isolation():
    ind = str_to_tree("(+ x (* x x))", CFG)
    c = reproduce(ind)
    c.children[0] = Node("-", [Node("x"), Node("x")])
    assert c != ind
    assert ind.children[0] == Node("x")


def test_probability_dispatch():
    # 固定种子多次抽样：reproduction（子代==亲本）与 crossover/mutation（子代!=亲本）
    # 均应出现
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


def test_apply_variation_single_parent_tail():
    """父本数为奇数时，最后一个未配对父本按原分发逻辑退化为变异/复制，数量守恒。

    基线行为（不额外消费随机数）：若抽到交叉但没有第二个父本，就走 mutation/reproduction。
    """
    parents = [str_to_tree("(+ x x)", CFG)] * 3
    saw_tail = False
    for seed in range(10):
        offspring, records = apply_variation_traced(parents, CFG, make_rng(seed))
        assert len(offspring) == 3
        assert sum(len(record.children) for record in records) == 3
        # 记录里的子代按顺序拼起来必须等于输出
        assert [c.sexpr_str() for r in records for c in r.children] == [
            c.sexpr_str() for c in offspring
        ]
        if records[0].event_type == "crossover":
            # 前两个父本配对后，剩下父本下标 2 单独处理，不可能是 crossover
            saw_tail = True
            assert len(records) == 2
            assert records[-1].parent_positions == (2,)
            assert records[-1].event_type in {"mutation", "reproduction"}
            assert len(records[-1].children) == 1
    assert saw_tail, "测试需要至少一个种子上出现“交叉 + 未配对尾部”的情形"


@pytest.mark.parametrize("bad", ["(x x)", "(+ x)", "(+ x x))", "(+ x", "y"])
def test_str_to_tree_rejects_bad_input(bad):
    with pytest.raises(ValueError):
        str_to_tree(bad, CFG)
