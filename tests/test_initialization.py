"""``gptoy.initialization`` 冒烟测试。"""

from gptoy.config import GPConfig
from gptoy.initialization import full, grow, initial_population
from gptoy.representation import Node
from gptoy.utils import make_rng

CFG = GPConfig()

TARGET_SEXPR = "(+ (+ (+ (* (* x x) (* x x)) (* x (* x x))) (* x x)) x)"


def _all_nodes(tree: Node):
    yield tree
    for child in tree.children:
        yield from _all_nodes(child)


def _leaf_depths(tree: Node, depth: int = 1):
    if not tree.children:
        return [depth]
    out = []
    for child in tree.children:
        out.extend(_leaf_depths(child, depth + 1))
    return out


def test_population_size():
    pop = initial_population(CFG, make_rng(0))
    assert len(pop) == CFG.population_size == 100


def test_node_sets():
    pop = initial_population(CFG, make_rng(0))
    allowed = set(CFG.function_set) | set(CFG.terminal_set)
    for tree in pop:
        for node in _all_nodes(tree):
            assert node.value in allowed


def test_depth_bounds():
    pop = initial_population(CFG, make_rng(0))
    for tree in pop:
        assert CFG.min_init_depth <= tree.depth() <= CFG.max_init_depth


def test_full_vs_grow_shapes():
    # full：所有叶子在同一深度，且深度 == 指定深度
    f = full(4, CFG, make_rng(1))
    assert f.depth() == 4
    leaves = _leaf_depths(f)
    assert len(set(leaves)) == 1 and leaves[0] == 4
    # grow：形状可变，深度在 [2, 指定深度] 内
    g = grow(4, CFG, make_rng(2))
    assert 2 <= g.depth() <= 4
    # 种群中同时存在深度不同的树（ramped 多样性）
    pop = initial_population(CFG, make_rng(3))
    depths = {t.depth() for t in pop}
    assert len(depths) >= 2


def test_no_target_injection():
    # 初始种群中不存在目标表达式（3.2 红线：不注入目标）
    pop = initial_population(CFG, make_rng(0))
    sexprs = {t.sexpr_str() for t in pop}
    assert TARGET_SEXPR not in sexprs


def test_deterministic_with_seed():
    a = [t.sexpr_str() for t in initial_population(CFG, make_rng(42))]
    b = [t.sexpr_str() for t in initial_population(CFG, make_rng(42))]
    assert a == b
