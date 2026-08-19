"""生成教学 notebook ``notebooks/koza1.ipynb``。

notebook 按 GP 主循环分 8 节，每节 = markdown 讲解 + 只调用 ``gptoy.*``
抽象算子的代码单元（Success Criteria 3.1：代码单元不内嵌任何一步的实现）。

随机种子取自 ``results/found_seed.json``（由 ``scripts/find_seed.py`` 生成），
也可以用 ``--seed`` 显式指定。

用法（在仓库根目录执行）：
    python scripts/build_notebook.py            # 使用 found_seed.json 中的种子
    python scripts/build_notebook.py --seed 42  # 指定种子（同时写入 found_seed.json）
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

RESULTS_DIR = ROOT / "results"
FOUND_SEED_PATH = RESULTS_DIR / "found_seed.json"
NOTEBOOK_PATH = ROOT / "notebooks" / "koza1.ipynb"

try:
    import nbformat
    from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook
except ImportError:  # pragma: no cover
    print("需要 nbformat：pip install nbformat", file=sys.stderr)
    sys.exit(1)


def _md(source: str):
    return new_markdown_cell(source.strip())


def _code(source: str):
    return new_code_cell(source.strip())


def build(seed: int) -> nbformat.notebooknode.NotebookNode:
    cells = []

    # =====================================================================
    # 头部
    # =====================================================================
    cells.append(_md(f"""
# Koza-1：用遗传编程从零恢复多项式（教学案例）

**问题**：只给 20 个数据点 $(x, y)$，其中 $y = f(x) = x^4 + x^3 + x^2 + x$，
但**不告诉**我们公式。用遗传编程（Genetic Programming, GP）从这 20 个点里
“猜出”这个表达式。

**本 notebook 的约定**（对应 `project.md` 的 Success Criteria）：

- 所有算子在 `gptoy` 包中按 GP 阶段**从零实现**（无任何遗传编程库）；
- 本 notebook 只做**编排与讲解**，每个代码单元只调用 `gptoy.*` 抽象算子；
- 固定随机种子 `SEED = {seed}`（由 `scripts/find_seed.py` 搜索得到，3.3 可复现）；
- 预算：种群 100、最多 50 代；目标：训练/泛化 MSE < 1e-6。

**目录**：① 问题与数据 → ② 个体表示 → ③ 初始化 → ④ 评价 → ⑤ 选择 →
⑥ 交叉与变异 → ⑦ 主循环 → ⑧ 结果与泛化。
"""))

    # =====================================================================
    # 第 1 节：问题与数据
    # =====================================================================
    cells.append(_md("""
## 1. 问题与数据

Koza-1 是遗传编程的经典入门问题：在 $[-1, 1]$ 上等距取 **20** 个点作为训练集，
目标是找到一个表达式，使它在这些点上的取值尽量接近 $y$。

- **训练集**：$[-1, 1]$ 等距 20 点 —— 进化过程只“看”这 20 个点；
- **泛化集**：$[1, 2]$ 等距 10 点 —— 最后用来检查学到的表达式是“真公式”还是
  只是把训练点插值了一遍（3.3 泛化验证）。

> 注意架构红线（3.2）：目标公式 $f(x)=x^4+x^3+x^2+x$ **只允许出现在
> `gptoy/data.py`**（用于生成标签）。评价函数只接收 `(tree, x, y)`，
> 永远看不到公式本身 —— 下面第 4 节会再强调这一点。
"""))
    cells.append(_code("""
# 1.1 固定种子 + 生成数据（唯一引入全局配置与数据的地方）
from gptoy.config import GPConfig
from gptoy.data import make_koza1_data
from gptoy.utils import set_global_seeds

SEED = 7  # 占位，构建时由 scripts/build_notebook.py 替换为 found_seed.json 的种子
set_global_seeds(SEED)          # 防御性固定全局随机种子（3.3 可复现）
cfg = GPConfig(seed=SEED)       # 全部超参数集中在此（种群 100 / 50 代 / 深度上限 17 ...）

x_train, y_train, x_test, y_test = make_koza1_data(cfg)
print(f"训练集：{x_train.shape[0]} 点 @ [{cfg.train_lo}, {cfg.train_hi}]")
print(f"泛化集：{x_test.shape[0]} 点 @ [{cfg.test_lo}, {cfg.test_hi}]")
print(f"训练标签前 5 个：{y_train[:5]}")
"""))
    cells.append(_code("""
# 1.2 看一眼训练数据：20 个点的分布
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(6, 3))
ax.plot(x_train, y_train, "o", label="training points (20)")
ax.plot(x_test, y_test, "s", label="generalization points (10)")
ax.set_xlabel("x"); ax.set_ylabel("y"); ax.legend(); ax.set_title("Koza-1 data")
"""))

    # =====================================================================
    # 第 2 节：个体表示
    # =====================================================================
    cells.append(_md("""
## 2. 个体表示：树

GP 里的一个“个体”就是一棵**表达式树**：

- **函数节点**：`+ - * /`，每个都有 **2** 个子节点（arity 2）；
- **终止符节点**：`x`（叶子，没有子节点；本项目终止符集只有 `x`，不含随机常数）。

一棵树 `(+ x (* x x))` 表示表达式 $x + x \\cdot x$。我们约定三种查看方式：

| 表示 | 例子 | 说明 |
|---|---|---|
| S-表达式 | `(+ x (* x x))` | 括号前缀式，程序内部标准形式 |
| 中缀 | `(x + (x * x))` | 数学书写习惯 |
| 结构量 | depth=3, size=5 | 深度 / 节点总数，限制树的大小用 |

`gptoy.representation` 提供树节点、结构量统计与字符串互转（`str_to_tree` 把
S-表达式字符串解析回树，方便我们手写例子）。
"""))
    cells.append(_code("""
# 2.1 用字符串解析构造示例树，并查看三种表示
from gptoy.representation import str_to_tree

t1 = str_to_tree("(+ x (* x x))", cfg)      # x + x*x
t2 = str_to_tree("(* (- x x) (/ x x))", cfg)  # (x - x) * (x / x)
for t in (t1, t2):
    print(f"S-表达式: {t.sexpr_str():20s} 中缀: {t.infix_str():22s} depth={t.depth()} size={t.size()}")
"""))
    cells.append(_code("""
# 2.2 深拷贝：GP 中“复制个体”必须互不影响
t1_copy = t1.deepcopy()
t1_copy.children[0] = str_to_tree("(- x x)", cfg)
print("原树不变：", t1.sexpr_str(), "| 副本已改：", t1_copy.sexpr_str())
"""))

    # =====================================================================
    # 第 3 节：初始化
    # =====================================================================
    cells.append(_md("""
## 3. 初始化：ramped half-and-half

初始种群必须**随机**生成（绝不能把目标表达式放进去，3.2 红线）。
Koza 经典做法是 **ramped half-and-half**：

- 深度桶：对 `min_init_depth..max_init_depth`（本项目 2..6）每个深度 `d` 生成一批树；
- 每个桶里**一半 full**：所有叶子都在同一深度 `d`（满树）；
- **一半 grow**：形状可变，深度不超过 `d`（随机提前停止生长）。

这样初始种群同时具备“深度多样性”（ramped）与“形状多样性”（full+grow），
为后面的选择/交叉/变异提供素材。
"""))
    cells.append(_code("""
# 3.1 生成初始种群：规模恰为 cfg.population_size = 100
from gptoy.initialization import initial_population
from gptoy.utils import make_rng

rng = make_rng(SEED)                      # 单一随机源（3.3 可复现）
pop0 = initial_population(cfg, rng)
print(f"初始种群规模: {len(pop0)}")
print("示例个体:", pop0[0].sexpr_str(), "| depth =", pop0[0].depth())
"""))
    cells.append(_code("""
# 3.2 初始种群的深度分布（ramped：每个深度桶都有树）
depths = [t.depth() for t in pop0]
fig, ax = plt.subplots(figsize=(5, 3))
ax.hist(depths, bins=range(cfg.min_init_depth, cfg.max_init_depth + 2), align="left", rwidth=0.8)
ax.set_xlabel("tree depth"); ax.set_ylabel("count"); ax.set_title("Initial population depth")
"""))

    # =====================================================================
    # 第 4 节：评价
    # =====================================================================
    cells.append(_md("""
## 4. 评价：递归求值 + MSE

个体好不好，要在训练集上算**适应度**。两步：

1. **递归求值** `evaluate(tree, x)`：按树后序递归 —— 终止符 `x` 直接返回值，
   函数节点先求两个子节点，再套用对应运算；
2. **保护除法**：除数为 0 时返回 1.0（Koza 经典约定），保证永远不会产生
   NaN/Inf 污染适应度 —— 例如 `(/ x x)` 在 $x=0$ 处求值 = 1.0。

适应度 = 训练集上的**均方误差 MSE**（越小越优）。注意 `fitness(tree, x, y)`
的签名：它只接收数据，**内部没有目标公式** —— 这正是 3.2 要求的
“不能把目标公式写死在评价函数里”。
"""))
    cells.append(_code("""
# 4.1 递归求值：标量与整批向量
from gptoy.operators import evaluate, fitness

print("t1 = x + x*x 在 x=3 处:", evaluate(t1, 3.0))
print("t1 在训练集前 5 个点上的值:", evaluate(t1, x_train[:5]))
"""))
    cells.append(_code("""
# 4.2 保护除法演示：x=0 时 x/x 返回 1.0，而不是 NaN
t_div = str_to_tree("(/ x x)", cfg)
print("(/ x x) 在 x=0 处:", evaluate(t_div, 0.0))
"""))
    cells.append(_code("""
# 4.3 评价整个初始种群：100 个 MSE（向量化求值，瞬间完成）
fits0 = [fitness(t, x_train, y_train) for t in pop0]
print(f"初始种群 best MSE = {min(fits0):.4f}   worst MSE = {max(fits0):.4f}")
"""))

    # =====================================================================
    # 第 5 节：选择
    # =====================================================================
    cells.append(_md("""
## 5. 选择：锦标赛选择

有了适应度就能“择优”。本项目用**锦标赛选择**（`k = tournament_size = 7`）：

- 从种群中**有放回**地随机抽 7 个个体，比 MSE，**最小者胜出**成为父本；
- 重复上述过程直到凑够父本数量；同一亲本可以多次胜出。

锦标赛的好处：实现简单、不需要处理负适应度、选择压力可由 `k` 调节。
（对比：适应度比例选择要求适应度非负且对异常值敏感 —— 这里先不展开。）

下面演示：从初始种群选 20 个父本，观察父本的 MSE 分布是否整体优于原种群
（这就是“选择压力”）。
"""))
    cells.append(_code("""
# 5.1 锦标赛选择：20 个父本
from gptoy.selection import tournament_select

parents = tournament_select(pop0, fits0, k=20, config=cfg, rng=make_rng(SEED))
print(f"选出父本数: {len(parents)}")
"""))
    cells.append(_code("""
# 5.2 选择压力：父本 MSE 分布 vs 原种群
parent_fits = [fitness(t, x_train, y_train) for t in parents]
print(f"原种群 MSE 均值: {sum(fits0)/len(fits0):.4f}   父本 MSE 均值: {sum(parent_fits)/len(parent_fits):.4f}")
"""))

    # =====================================================================
    # 第 6 节：交叉与变异
    # =====================================================================
    cells.append(_md("""
## 6. 交叉与变异

新个体由三种**变异算子**生成（`gptoy.variation`）：

| 算子 | 做法 | 深度控制 |
|---|---|---|
| `crossover(p1, p2)` | 各随机选一个交叉点，交换两棵子树 | 子代深度 > 17 则重选交叉点，仍超限则回退为亲本副本 |
| `mutate(ind)` | 随机选一个节点，把该子树替换为 grow 新树 | 新子树深度受“剩余深度预算”约束 |
| `reproduce(ind)` | 深拷贝原个体 | — |

`apply_variation` 按概率分发三种算子：交叉 0.7 / 变异 0.2 / 复制 0.1 ——
保证一轮运行中三种算子都可观测（3.1）。精英保留（每代 1 棵最优）在主循环里做。
"""))
    cells.append(_code("""
# 6.1 交叉：交换子树
from gptoy.variation import apply_variation, crossover, mutate, reproduce

p1, p2 = pop0[0], pop0[1]
c1, c2 = crossover(p1, p2, cfg, make_rng(SEED))
print("p1:", p1.sexpr_str())
print("p2:", p2.sexpr_str())
print("c1:", c1.sexpr_str(), "depth =", c1.depth())
print("c2:", c2.sexpr_str(), "depth =", c2.depth())
"""))
    cells.append(_code("""
# 6.2 变异与复制
m = mutate(p1, cfg, make_rng(SEED))
r = reproduce(p1)
print("变异后:", m.sexpr_str(), "depth =", m.depth())
print("复制后与亲本结构相等:", r == p1, "| 但两者互不影响（深拷贝）")
"""))
    cells.append(_code("""
# 6.3 概率分发：把 99 个父本变成 99 个下一代个体
offspring = apply_variation(parents, cfg, make_rng(SEED))
print(f"apply_variation 输出 {len(offspring)} 个个体（与父本数量一致）")
"""))

    # =====================================================================
    # 第 7 节：主循环
    # =====================================================================
    cells.append(_md("""
## 7. 主循环：生成式进化 + 精英保留

把前面所有算子串起来就是 GP 主循环（`gptoy.evolution.run_gp`），每一代：

1. 评价当前种群（MSE）；
2. 记录**历代最优** best-of-run 与收敛历史；
3. **精英保留** 1 棵最优个体直接进入下一代；
4. **锦标赛选择**出 99 个父本；
5. **交叉/变异/复制**生成 99 个新个体 → 与精英合并成下一代（种群恒为 100）。

预算：最多 `max_generations = 50` 代；一旦 best MSE < `mse_target = 1e-6`
就**提前收敛**停止（省时间，教学上也能看到“进化到第几代成功”）。

> 下面这一行就是整个算法。运行时间通常只有几秒（20 点向量化求值 + 提前收敛）。
"""))
    cells.append(_code("""
# 7.1 跑完整 GP：初始化 → 逐代（选择/交叉/变异/精英）→ 收敛
from gptoy.evolution import run_gp

result = run_gp(cfg)   # 内部用 make_rng(cfg.seed) 作为唯一随机源，逐位可复现
print(f"第 0 代 best MSE : {result.gen0_mse:.6f}   （> 1e-3，说明不是第 0 代碰巧解出）")
print(f"收敛代数         : {result.generation_found}")
print(f"实际运行代数     : {result.n_generations}  （预算 {cfg.max_generations}）")
print(f"训练 MSE         : {result.train_mse:.3e}")
print(f"泛化 MSE         : {result.test_mse:.3e}")
print(f"运行耗时         : {result.total_seconds:.2f} s")
"""))

    # =====================================================================
    # 第 8 节：结果与泛化
    # =====================================================================
    cells.append(_md("""
## 8. 结果与泛化

**收敛曲线**：横轴是代数，纵轴是每代 best MSE（对数坐标）。
曲线单调不增是精英保留的保证，台阶处就是“发现更好个体”的时刻。

**泛化检查**（3.3 的关键）：如果算法只是在 20 个训练点上插值，那么在
$[1, 2]$ 这 10 个**没见过**的点上 MSE 会很大。因此要求：

> 训练 MSE < 1e-6 **且** 泛化 MSE < 1e-6 —— 才算真正恢复了 $f(x)=x^4+x^3+x^2+x$。

最后把学到的树画出来与真实曲线对比（真实曲线来自 `gptoy.data.koza1_target`，
它是数据模块的公开接口，用于展示，不参与评价）。
"""))
    cells.append(_code("""
# 8.1 收敛曲线
fig, ax = plt.subplots(figsize=(6, 3))
ax.semilogy(result.best_mse_history, "-o", ms=3)
ax.axhline(cfg.mse_target, color="r", ls="--", lw=1, label="target 1e-6")
ax.set_xlabel("generation"); ax.set_ylabel("best MSE (log)"); ax.legend()
ax.set_title("Convergence curve")
"""))
    cells.append(_code("""
# 8.2 最终表达式
best = result.best_of_run
print("S-表达式:", best.sexpr_str())
print("中缀形式:", best.infix_str())
"""))
    cells.append(_code("""
# 8.3 学到的函数 vs 真实曲线（训练区间与泛化区间都画）
# 真实曲线来自 gptoy.data.koza1_target —— 数据模块的公开接口，仅用于展示，不参与评价
import numpy as np
from gptoy.data import koza1_target
from gptoy.operators import evaluate

xs_dense = np.linspace(-1.0, 2.0, 200)
fig, ax = plt.subplots(figsize=(6, 3.5))
ax.plot(xs_dense, evaluate(best, xs_dense), label="learned tree")
ax.plot(xs_dense, koza1_target(xs_dense), "--", lw=1, label="true f(x)")
ax.plot(x_train, y_train, "o", ms=4, label="training points")
ax.plot(x_test, y_test, "s", ms=4, label="generalization points")
ax.axvline(1.0, color="gray", ls=":", lw=1)
ax.set_xlabel("x"); ax.set_ylabel("y"); ax.legend()
ax.set_title("Learned function vs target (dashed line: train/generalize boundary)")
"""))
    cells.append(_code("""
# 8.4 对照 Success Criteria 3.3 逐条判定
ok_train = result.train_mse < cfg.mse_target
ok_test = result.test_mse < cfg.mse_target
ok_gen0 = result.gen0_mse > cfg.gen0_unsolved_mse
print(f"[3.3] 训练 MSE {result.train_mse:.3e} < 1e-6 : {'PASS' if ok_train else 'FAIL'}")
print(f"[3.3] 泛化 MSE {result.test_mse:.3e} < 1e-6 : {'PASS' if ok_test else 'FAIL'}")
print(f"[3.1] 第 0 代未解出 (gen0 MSE {result.gen0_mse:.4f} > 1e-3) : {'PASS' if ok_gen0 else 'FAIL'}")
print(f"[3.1] 预算内 ({result.n_generations} <= {cfg.max_generations} 代) : "
      f"{'PASS' if result.n_generations <= cfg.max_generations else 'FAIL'}")
print("结论：", "全部 PASS —— 在约定预算内从 20 个点恢复了目标多项式。" if all([ok_train, ok_test, ok_gen0]) else "未达标，请重新搜索种子。")
"""))

    nb = new_notebook(
        cells=cells,
        metadata={
            "kernelspec": {
                "display_name": "Python 3 (ipykernel)",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.12"},
        },
    )
    return nb


def _inject_seed(nb, seed: int) -> None:
    """把真实种子替换进 1.1 代码单元中的占位符。"""
    replaced = 0
    for cell in nb.cells:
        if cell.cell_type != "code":
            continue
        if "SEED = 7  # 占位" in cell.source:
            cell.source = cell.source.replace(
                "SEED = 7  # 占位",
                f"SEED = {seed}  # 由 scripts/find_seed.py 搜索得到（3.3 可复现）",
            )
            replaced += 1
    if replaced != 1:
        raise RuntimeError("notebook 中未找到 SEED 占位符")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=None, help="显式指定种子（覆盖 found_seed.json）")
    args = parser.parse_args()

    if args.seed is not None:
        seed = args.seed
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        FOUND_SEED_PATH.write_text(
            json.dumps({"seed": seed, "source": "build_notebook.py --seed"}, indent=2),
            encoding="utf-8",
        )
    elif FOUND_SEED_PATH.exists():
        seed = int(json.loads(FOUND_SEED_PATH.read_text(encoding="utf-8"))["seed"])
    else:
        print("未找到 results/found_seed.json，请先运行 scripts/find_seed.py（或用 --seed 指定）",
              file=sys.stderr)
        return 1

    nb = build(seed)
    _inject_seed(nb, seed)
    NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(nb, NOTEBOOK_PATH)
    print(f"已生成 {NOTEBOOK_PATH}（SEED = {seed}，{len(nb.cells)} 个单元）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
