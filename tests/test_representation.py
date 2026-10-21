"""``gptoy.representation`` 测试：结构、路径访问、结构相等与哈希一致。"""

import pytest

from gptoy.config import GPConfig
from gptoy.representation import Node, str_to_tree

CFG = GPConfig()


def test_node_construction():
    # 函数节点：arity 2 合法
    n = Node("+", [Node("x"), Node("x")])
    assert n.value == "+" and len(n.children) == 2
    # 非法 arity 报错
    with pytest.raises(ValueError):
        Node("+", [Node("x")])
    # 非法节点集合报错
    with pytest.raises(ValueError):
        Node("%", [Node("x"), Node("x")])
    # 终止符不允许有子节点
    with pytest.raises(ValueError):
        Node("x", [Node("x")])


def test_depth_and_size():
    t = str_to_tree("(+ x (* x x))", CFG)
    assert t.depth() == 3
    assert t.size() == 5
    assert str_to_tree("x", CFG).depth() == 1
    assert str_to_tree("x", CFG).size() == 1


def test_sexpr_roundtrip():
    src = "(+ (* x x) (- (/ x x) x))"
    t = str_to_tree(src, CFG)
    assert str_to_tree(t.sexpr_str(), CFG) == t


def test_infix_str():
    t = str_to_tree("(+ x (* x x))", CFG)
    assert t.infix_str() == "(x + (x * x))"
    assert str_to_tree("x", CFG).infix_str() == "x"


def test_deepcopy_independent():
    t = str_to_tree("(+ x (* x x))", CFG)
    c = t.deepcopy()
    c.children[0] = Node("-", [Node("x"), Node("x")])
    assert c != t
    assert t.children[0] == Node("x")


def test_structural_equality_and_hash_consistent():
    """结构相同的两棵不同对象必须相等、同 hash，并能被 set 去重。"""
    a = str_to_tree("(+ x (* x x))", CFG)
    b = str_to_tree("(+ x (* x x))", CFG)
    assert a is not b
    assert a == b
    assert hash(a) == hash(b)
    assert len({a, b}) == 1

    c = str_to_tree("(+ (* x x) x)", CFG)  # 函数相同但结构不同
    assert a != c
    assert len({a, b, c}) == 2

    # 终止符同样遵循结构相等
    assert Node("x") == Node("x")
    assert hash(Node("x")) == hash(Node("x"))
    assert len({Node("x"), Node("x")}) == 1


def test_hash_is_not_identity_based():
    # 若 hash 基于对象身份，10 棵结构相同的树会得到 10 个不同 hash/10 个集合元素
    trees = [str_to_tree("(* (+ x x) (- x x))", CFG) for _ in range(10)]
    assert len({hash(t) for t in trees}) == 1
    assert len(set(trees)) == 1


def test_equality_with_other_types():
    t = str_to_tree("(+ x x)", CFG)
    assert (t == "(+ x x)") is False
    assert t != 3


def test_iter_nodes_paths():
    t = str_to_tree("(+ x (* x x))", CFG)
    pairs = list(t.iter_nodes())
    # 先序遍历：根 → 左子树 → 右子树
    assert [path for _, path in pairs] == [(), (0,), (1,), (1, 0), (1, 1)]
    assert [node.value for node, _ in pairs] == ["+", "x", "*", "x", "x"]
    assert t.subtree_at((1, 0)).value == "x"
    with pytest.raises(IndexError):
        t.subtree_at((5,))


def test_replace_subtree_returns_new_tree():
    t = str_to_tree("(+ x (* x x))", CFG)
    new = t.replace_subtree((0,), str_to_tree("(- x x)", CFG))
    assert new.sexpr_str() == "(+ (- x x) (* x x))"
    assert t.sexpr_str() == "(+ x (* x x))"  # 原树不变
    # 替换根节点
    assert t.replace_subtree((), Node("x")).sexpr_str() == "x"


def test_str_to_tree_errors():
    with pytest.raises(ValueError):
        str_to_tree("(+ x)", CFG)  # 缺第二个子节点
    with pytest.raises(ValueError):
        str_to_tree("(+ x x", CFG)  # 缺右括号
    with pytest.raises(ValueError):
        str_to_tree("(x x)", CFG)  # 终止符作运算符
