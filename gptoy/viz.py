"""教学可视化辅助：表达式树绘制。

只负责“把一棵已经存在的树画出来”，不参与 GP 算法。Notebook 用它展示
交叉/变异前后的树、高亮交叉点/变异点路径。

``draw_tree`` 的坐标约定：叶子从左到右依次占据整数 x，内部节点的 x 取子节点
x 的均值；y = 深度的负值（根在上）。
"""

from __future__ import annotations

from .representation import Node


def tree_layout(root: Node) -> dict[int, tuple[float, float]]:
    """计算树的绘制坐标，返回 ``{id(node): (x, y)}``。

    使用 ``id(node)`` 而不是 ``node`` 作为键：``Node`` 是可变的，按对象身份索引
    可以避免“绘制期间不改树”之外的哈希约束。
    """
    positions: dict[int, tuple[float, float]] = {}
    next_leaf_x = 0.0

    def visit(node: Node, depth: int) -> float:
        nonlocal next_leaf_x
        if not node.children:
            x = next_leaf_x
            next_leaf_x += 1.0
        else:
            child_xs = [visit(child, depth + 1) for child in node.children]
            x = sum(child_xs) / len(child_xs)
        positions[id(node)] = (x, -float(depth))
        return x

    visit(root, 0)
    return positions


def _resolve_path(root: Node, path: tuple[int, ...]) -> list[int]:
    """把路径展开成从根到目标的节点 id 列表（用于高亮）。"""
    ids = [id(root)]
    node = root
    for index in path:
        node = node.children[index]
        ids.append(id(node))
    return ids


def draw_tree(
    ax,
    root: Node,
    *,
    title: str | None = None,
    highlight_paths=(),
    highlight_color: str = "#d62728",
    node_color: str = "#4c72b0",
    show_axis: bool = False,
):
    """在 ``ax`` 上绘制表达式树。

    参数
    ----
    root : Node
        要画的树。
    highlight_paths : iterable[tuple[int, ...]]
        需要高亮的路径集合（如交叉点 / 变异点）；高亮节点与路径边会变色。
    """
    positions = tree_layout(root)
    highlighted = set()
    for path in highlight_paths:
        highlighted.update(_resolve_path(root, path))

    # 先画边，再画节点，保证节点在上层
    for node, _ in root.iter_nodes():
        x0, y0 = positions[id(node)]
        for child in node.children:
            x1, y1 = positions[id(child)]
            on_path = id(node) in highlighted and id(child) in highlighted
            ax.plot(
                [x0, x1],
                [y0, y1],
                color=highlight_color if on_path else "#888888",
                lw=2.4 if on_path else 1.2,
                zorder=1,
            )

    for node, _ in root.iter_nodes():
        x, y = positions[id(node)]
        color = highlight_color if id(node) in highlighted else node_color
        marker = "s" if node.children else "o"
        ax.scatter([x], [y], s=460, color=color, marker=marker, zorder=2, edgecolors="white")
        ax.text(
            x,
            y,
            node.value,
            ha="center",
            va="center",
            color="white",
            fontsize=9,
            fontweight="bold",
            zorder=3,
        )

    xs = [p[0] for p in positions.values()]
    ys = [p[1] for p in positions.values()]
    ax.set_xlim(min(xs) - 0.6, max(xs) + 0.6)
    ax.set_ylim(min(ys) - 0.7, max(ys) + 0.7)
    if title:
        ax.set_title(title)
    if not show_axis:
        ax.set_axis_off()


def plot_tree(root: Node, *, title: str | None = None, highlight_paths=(), figsize=(7, 3.2)):
    """创建新 figure 并绘制一棵树，返回 ``(fig, ax)``。"""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=figsize)
    draw_tree(ax, root, title=title, highlight_paths=highlight_paths)
    fig.tight_layout()
    return fig, ax
