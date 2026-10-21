"""个体表示：表达式树（``Node``）。

- 函数节点：``{"+", "-", "*", "/"}``，arity 2
- 终止符节点：``{"x"}``（Koza-1 的终止符集只有 x，不含随机常数）

提供深度/规模统计、中缀与 S-表达式显示、路径式结构访问（``iter_nodes`` /
``subtree_at`` / ``replace_subtree``）以及 ``str_to_tree``（测试与 notebook 展示用）。

结构相等与哈希
--------------
``__eq__`` 是**结构相等**（同一位置、同一节点值、同构子树），``__hash__`` 也按
结构计算，因此 ``a == b`` 蕴含 ``hash(a) == hash(b)``，可以把 ``Node`` 放进
``set`` / ``dict`` 统计结构唯一性。注意 ``Node`` 是可变对象：一旦把某棵树放进
``set`` / ``dict``，就不要再原地修改它（``variation`` 中的替换都在深拷贝上进行）。
"""

from __future__ import annotations

from typing import Iterator

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

    # ---- 结构访问（按路径）----
    def iter_nodes(self) -> Iterator[tuple["Node", tuple[int, ...]]]:
        """先序遍历整棵树，产出 ``(节点, 根到该节点的路径)``。

        路径是子节点下标组成的元组；根的路径为空元组 ``()``。遍历顺序固定为
        「根 → 左子树 → 右子树」，是 ``variation`` 选交叉点/变异点的唯一定义。
        """
        yield self, ()
        for index, child in enumerate(self.children):
            for node, path in child.iter_nodes():
                yield node, (index,) + path

    def subtree_at(self, path: tuple[int, ...]) -> "Node":
        """返回路径指向的子树（不复制；越界抛 ``IndexError``）。"""
        node: Node = self
        for index in path:
            node = node.children[index]
        return node

    def replace_subtree(self, path: tuple[int, ...], new_subtree: "Node") -> "Node":
        """返回一棵新树：深拷贝 ``self`` 后把 ``path`` 处子树替换为 ``new_subtree``。

        原始树不被修改；新子树同样被深拷贝，因此结果与两个输入都无共享节点。
        """
        if not path:
            return new_subtree.deepcopy()
        new_root = self.deepcopy()
        node = new_root
        for index in path[:-1]:
            node = node.children[index]
        node.children[path[-1]] = new_subtree.deepcopy()
        return new_root

    # ---- 显示 ----
    def infix_str(self) -> str:
        """中缀字符串，如 ``"(x + x)"``。"""
        if not self.children:
            return self.value
        return f"({self.children[0].infix_str()} {self.value} {self.children[1].infix_str()})"

    def sexpr_str(self) -> str:
        """S-表达式字符串，如 ``"(+ x x)"``。

        在本表示中 S-表达式是树的**规范签名**：结构相同的树必然得到相同字符串，
        结构不同的树得到不同字符串（``metrics.structural_signature`` 直接复用它）。
        """
        if not self.children:
            return self.value
        inner = " ".join(child.sexpr_str() for child in self.children)
        return f"({self.value} {inner})"

    # ---- 相等、哈希与复制 ----
    def __eq__(self, other) -> bool:
        """结构相等（递归比较节点值与子树）。"""
        if not isinstance(other, Node):
            return NotImplemented
        if self.value != other.value or len(self.children) != len(other.children):
            return False
        return all(a == b for a, b in zip(self.children, other.children))

    def __hash__(self) -> int:
        """结构哈希：与 ``__eq__`` 保持一致（结构相同的树 hash 相同）。"""
        return hash((self.value, tuple(self.children)))

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
