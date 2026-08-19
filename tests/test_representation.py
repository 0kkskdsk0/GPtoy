"""``gptoy.representation`` 冒烟测试（验证器从简：只验证语义正确性）。"""

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


def test_str_to_tree_errors():
    with pytest.raises(ValueError):
        str_to_tree("(+ x)", CFG)  # 缺第二个子节点
    with pytest.raises(ValueError):
        str_to_tree("(+ x x", CFG)  # 缺右括号
    with pytest.raises(ValueError):
        str_to_tree("(x x)", CFG)  # 终止符作运算符
