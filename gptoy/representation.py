"""个体表示：树（``Node``）。

- 函数节点：``{"+", "-", "*", "/"}``，arity 2
- 终止符节点：``{"x"}``（3.2：终止符集只有 x，不含随机常数）

提供深度/规模统计、中缀与 S-表达式显示、``str_to_tree``（冒烟测试与 notebook 展示用）。
"""

from __future__ import annotations

from .config import GPConfig

# 节点集合校验用默认配置（不参与随机性，仅为结构合法性检查）。
_DEFAULT_CONFIG = GPConfig()

# 函数节点 arity（本项目函数集全为二元运算）。
_FUNCTION_ARITY = 2


class Node:
    """GP 树节点。

    ``value``：str —— 运算符（如 ``"+"``）或终止符（如 ``"x"``）
    ``children``：``list[Node]`` —— 函数节点的子节点；终止符节点为空列表
    """

    def __init__(self, value: str, children: list["Node"] | None = None):
        children = children if children is not None else []
        if value in _DEFAULT_CONFIG.function_set:
            if len(children) != _FUNCTION_ARITY:
                raise ValueError(
                    f"函数节点 {value!r} 需要 {_FUNCTION_ARITY} 个子节点，实际 {len(children)}"
                )
        elif value in _DEFAULT_CONFIG.terminal_set:
            if children:
                raise ValueError(f"终止符节点 {value!r} 不允许有子节点")
        else:
            raise ValueError(
                f"非法节点值 {value!r}：必须属于函数集 {_DEFAULT_CONFIG.function_set} "
                f"或终止符集 {_DEFAULT_CONFIG.terminal_set}"
            )
        self.value = value
        self.children = list(children)

    # ---- 结构量 ----
    def depth(self) -> int:
        """树深度：单节点为 1，逐层向下计数。"""
        if not self.children:
            return 1
        return 1 + max(child.depth() for child in self.children)

    def size(self) -> int:
        """节点总数（个体规模）。"""
        return 1 + sum(child.size() for child in self.children)

    # ---- 显示 ----
    def infix_str(self) -> str:
        """中缀字符串，如 ``"(x + x)"``。"""
        if not self.children:
            return self.value
        return f"({self.children[0].infix_str()} {self.value} {self.children[1].infix_str()})"

    def sexpr_str(self) -> str:
        """S-表达式字符串，如 ``"(+ x x)"``。"""
        if not self.children:
            return self.value
        inner = " ".join(child.sexpr_str() for child in self.children)
        return f"({self.value} {inner})"

    # ---- 相等与复制 ----
    def __eq__(self, other) -> bool:
        """结构相等（递归比较）。"""
        if not isinstance(other, Node):
            return NotImplemented
        if self.value != other.value or len(self.children) != len(other.children):
            return False
        return all(a == b for a, b in zip(self.children, other.children))

    def __hash__(self) -> int:
        # __eq__ 被定义后显式标记不可哈希（树用于集合比较时走 __eq__）。
        return object.__hash__(self)

    def __repr__(self) -> str:
        return f"Node({self.sexpr_str()!r})"

    def deepcopy(self) -> "Node":
        """深拷贝，保证与原始树完全独立。"""
        return Node(self.value, [child.deepcopy() for child in self.children])


def _tokenize(s: str) -> list[str]:
    """把 S-表达式字符串切分为 token 列表（括号、运算符、终止符）。"""
    return s.replace("(", " ( ").replace(")", " ) ").split()


def _parse_tokens(tokens: list[str], pos: int, config: GPConfig):
    """递归解析 token 列表，返回 ``(node, 下一个位置)``。"""
    if pos >= len(tokens):
        raise ValueError("S-表达式不完整：意外到达末尾")
    tok = tokens[pos]
    if tok == "(":
        # 列表形式：(op child1 child2 ...)
        if pos + 1 >= len(tokens) or tokens[pos + 1] == ")":
            raise ValueError(f"S-表达式非法：位置 {pos} 处缺少运算符")
        op = tokens[pos + 1]
        if op in config.terminal_set:
            raise ValueError(f"终止符 {op!r} 不应带子节点")
        if op not in config.function_set:
            raise ValueError(f"非法运算符 {op!r}")
        children = []
        p = pos + 2
        while p < len(tokens) and tokens[p] != ")":
            child, p = _parse_tokens(tokens, p, config)
            children.append(child)
        if p >= len(tokens):
            raise ValueError("S-表达式非法：缺少右括号")
        if len(children) != _FUNCTION_ARITY:
            raise ValueError(f"运算符 {op!r} 需要 {_FUNCTION_ARITY} 个子节点，实际 {len(children)}")
        return Node(op, children), p + 1
    if tok == ")":
        raise ValueError("S-表达式非法：多余的右括号")
    if tok in config.terminal_set:
        return Node(tok), pos + 1
    raise ValueError(f"非法终止符 {tok!r}")


def str_to_tree(s: str, config: GPConfig) -> Node:
    """把 S-表达式字符串解析回树（冒烟测试与 notebook 示例用）。

    例：``str_to_tree("(+ x (* x x))", config)``
    """
    tokens = _tokenize(s)
    if not tokens:
        raise ValueError("空表达式")
    node, pos = _parse_tokens(tokens, 0, config)
    if pos != len(tokens):
        raise ValueError(f"S-表达式非法：位置 {pos} 处有多余内容")
    return node
