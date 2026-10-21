"""GPtoy 学习实验室的内容定义：Notebook 单元 + 笔记骨架元数据。

本模块只存放教学内容（Markdown 文本与可运行代码单元），由
``scripts/build_learning_lab.py`` 组装成 ``notebooks/GP_learning_lab.ipynb``
与 ``notes/GP_learning_notes.md``。

约定
----
- 每个章节的标题固定为 ``## Chapter N：标题``，Notebook 与笔记共用；
- 每个实验标签 ``E01``–``E10`` 同时出现在讲解 Markdown 与代码单元注释
  （``# E01``）中，便于自动检查“实验有真实运行入口”；
- 所有需要学习者填写的区域只放占位符（``我的预测（请填写）`` /
  ``观察与解释（请填写）`` / ``待填写``），绝不预填个人结论。
"""

import textwrap


def md(text: str) -> tuple[str, str]:
    return ("md", textwrap.dedent(text).strip("\n"))


def code(text: str) -> tuple[str, str]:
    return ("code", textwrap.dedent(text).strip("\n"))


# ---------------------------------------------------------------------------
# 章节元数据：Notebook 与笔记骨架共用，保证编号、标题、核心问题完全一致
# ---------------------------------------------------------------------------
PARTS = [
    ("第一篇", "GP 的经典运行机制", "Chapter 0–6：从表示、初始化、评价、选择、变异到完整进化循环"),
    ("第二篇", "GP 的典型失败", "Chapter 7–10：随机性与预算、早熟与膨胀、噪声与泛化、空间与数值限制"),
    ("第三篇", "从实验走向研究", "Chapter 11：把观察组织成可检验的失败机制与改进方向"),
]

CHAPTERS = [
    {
        "num": 0,
        "title": "GP 与符号回归",
        "part": 0,
        "core_question": "GP 在符号回归里到底搜索什么？训练/测试 MSE 很小，能说明公式被严格恢复了吗？",
        "source": "`gptoy/data.py`（`koza1_target`、`make_koza1_data`）、`gptoy/operators.py`（`fitness`）、`gptoy/evolution.py`（`run_gp`）",
        "experiments": [],
        "concepts": [
            "符号回归：只给有限个 $(x_i, y_i)$，在一个由函数集与终止符集递归定义的表达式空间中搜索",
            "搜索空间：所有合法的表达式树；结构发现（选哪些运算、怎样嵌套）与常数优化（连续参数）是两类不同问题",
            "Koza-1：训练集 $[-1,1]$ 等距 20 点，测试/泛化集 $[1,2]$ 等距 10 点，目标 $f(x)=x^4+x^3+x^2+x$ 只用于生成标签",
            "三级区分：数值拟合（训练/测试 MSE）< 函数一致（在讨论的区间上逐点一致）< 结构恢复（结构与某个等价推导相同）",
            "小 MSE 只能说明在有限样本上拟合好，不能自动证明真实公式被严格恢复",
        ],
    },
    {
        "num": 1,
        "title": "表达式树与程序表示",
        "part": 0,
        "core_question": "一个 GP 个体怎样被表示？不同的树什么时候表示同一个函数？保护运算改变了什么语义？",
        "source": "`gptoy/representation.py`（`Node`、`iter_nodes`、`subtree_at`、`replace_subtree`、`str_to_tree`）、`gptoy/operators.py`（`evaluate`、`p_div`）、`gptoy/viz.py`（`draw_tree`）",
        "experiments": ["E01"],
        "concepts": [
            "Function Set $\\{+,-,*,/\\}$、Terminal Set $\\{x\\}$；函数节点 arity 2，终止符是叶子",
            "语法树、genotype/phenotype：本项目里基因型与表现型是同一棵树（树就是程序）",
            "深度 depth（单节点为 1）与规模 size（节点总数）；满二叉树深度 $d$ 的节点数是 $2^d-1$",
            "语法（S-表达式）唯一确定结构；不同结构可以表示同一个数学函数",
            "保护除法：分母精确为 0 时返回 1.0，这会改变数学语义，使部分代数恒等式不再全域成立",
        ],
    },
    {
        "num": 2,
        "title": "种群初始化",
        "part": 0,
        "core_question": "初始种群怎样生成？full / grow / ramped half-and-half 会带来怎样的搜索起点与偏差？",
        "source": "`gptoy/initialization.py`（`full`、`grow`、`initial_population`、`INIT_METHODS`）",
        "experiments": ["E02"],
        "concepts": [
            "Population：一代中所有候选个体；初始化决定搜索的起点分布",
            "Full：所有叶子在同一深度，形状规则的满树；深度 $d$ 时节点数固定为 $2^d-1$",
            "Grow：不超过给定深度、可提前停止生长，形状与深度都可变",
            "Ramped half-and-half：按深度分桶，每个桶一半 full 一半 grow，再打乱顺序",
            "初始化偏差：不同方法覆盖的深度/形状比例不同，会影响后续选择与变异的可及区域",
        ],
    },
    {
        "num": 3,
        "title": "适应度评价",
        "part": 0,
        "core_question": "适应度怎样定义？MSE 能评价什么、不能评价什么？训练/验证/测试各自负责什么？",
        "source": "`gptoy/operators.py`（`fitness`、`evaluate`）、`gptoy/data.py`（`make_koza1_data`）、`gptoy/metrics.py`（`complexity`）",
        "experiments": ["E03"],
        "concepts": [
            "经验风险：$\\mathrm{MSE}(T)=\\frac{1}{n}\\sum_i (T(x_i)-y_i)^2$，是搜索时唯一可见的信号",
            "结构复杂度：`size`（节点数）、`depth`（深度）；同样 MSE 可以有不同复杂度",
            "训练集用于搜索；验证集可用于模型/超参数选择；测试集只做最终评价，不反馈",
            "经验拟合 ≠ 结构恢复：有限样本上的小 MSE 不能区分“真结构”与“碰巧在样本上一致的结构”",
            "欠定问题：训练点太少时，多个不同结构可以有相同的训练误差",
        ],
    },
]


# ---------------------------------------------------------------------------
# 前置部分
# ---------------------------------------------------------------------------
def front_matter() -> list:
    cells = [
        md(r"""
            # GPtoy 学习实验室：从零理解遗传编程

            这是一个围绕 **Koza-1 符号回归** 的系统化 GP 学习实验室。目标不是把 GP 调得更快或更强，
            而是让你能回答下面五个问题，并能用**可复现的实验**支持自己的回答：

            1. GP 的各个阶段分别执行了什么操作？
            2. 每个阶段为什么需要这样的设计？
            3. 结构、参数与随机性如何共同决定搜索过程？
            4. GP 为什么有时成功、有时失败？
            5. 怎样用对照实验区分不同类型的失败，并把现象变成可调研的研究问题？

            ## 学习方法：预测 → 运行 → 观察 → 解释

            每个实验都按同一协议组织，请**先写预测再运行**：

            | 步骤 | 你要做的事 |
            |---|---|
            | 1. 预测 | 在实验的 `我的预测（请填写）` 处写下你的判断与理由 |
            | 2. 运行 | 直接运行代码单元；如需改动，修改单元顶部标出的参数后重跑 |
            | 3. 观察 | 记录真实数字/图形，区分“看到了什么”和“我以为什么” |
            | 4. 解释 | 写 `观察与解释（请填写）`：支持/反对证据、替代解释、证据边界 |

            **本 Notebook 只提供可复现的实验与数据，不替你写研究结论。** 所有个人区域都是空占位。

            ## 环境与使用方法

            - Python 3.12，依赖 `numpy`、`matplotlib`；Jupyter kernel 名为 **`gptoy`**；
            - 所有算法在 `gptoy/` 包中从零实现，Notebook 只负责讲解、调用与可视化；
            - 无网络、无 `input()`、无需手动点击；从头执行即可复现全部输出；
            - 若图不显示：确认 kernel 是 `gptoy`，并检查单元 `%matplotlib inline` 已执行；
            - 默认配置刻意很小（种群 20–100、代数 5–50），完整跑完通常几十秒；
              放大方法见文末 **附录 A**。

            ## 目录

            - **第一篇：GP 的经典运行机制**
              - Chapter 0：GP 与符号回归（基线运行）
              - Chapter 1：表达式树与程序表示（E01）
              - Chapter 2：种群初始化（E02）
              - Chapter 3：适应度评价（E03）
              - Chapter 4：选择机制（E04）
              - Chapter 5：遗传操作（E05）
              - Chapter 6：完整进化过程（E06）
            - **第二篇：GP 的典型失败**
              - Chapter 7：随机性与搜索预算（E07）
              - Chapter 8：早熟收敛与表达式膨胀（E08）
              - Chapter 9：小样本、噪声与泛化（E09）
              - Chapter 10：搜索空间与数值语义的限制（E10）
            - **第三篇：从实验走向研究**
              - Chapter 11：GP 失败机制与改进方法（综合 E01–E10）
            - **附录**：保存实验结果、放大实验规模、保存个人笔记、常见问题

            ## 记号与约定

            | 记号 | 含义 |
            |---|---|
            | `mse_target = 1e-6` | “训练/测试数值达标”的阈值，**不是**“公式被严格恢复”的判据 |
            | `best_fitness` | 当代最小适应度（本项目 MSE 越小越好） |
            | `generation 0` | 初始种群这一代（尚未发生任何遗传操作） |
            | `evaluations` | 为搜索而做的训练适应度评估次数（测试评价不计入） |
            | 测试集 | 只用于最终评价；不参与选择、调参或挑选种子 |

            ## 科学诚实：四级证据标签

            本 Notebook 的叙述尽量显式区分四类陈述，请你也用同样的标签写笔记：

            - **观察**：本次运行实际打印/画出的数字与图形；
            - **机制解释**：由对照实验支持的因果解释（说明控制了哪些变量）；
            - **猜测**：尚未检验的推测，必须标注为猜测；
            - **文献/数学事实**：来自外部知识或数学证明，须注明依据（如“sin 不是有理函数”）。

            把数值拟合直接写成“严格恢复了公式”是本任务明确禁止的过度声明。
        """),
        code(r"""
            %matplotlib inline
            # 环境检查与公共导入：所有实验都只通过 gptoy 包调用算法
            import sys
            from dataclasses import replace
            from pathlib import Path

            import numpy as np
            import matplotlib
            import matplotlib.pyplot as plt

            # 保证 notebook 在内核工作目录为仓库根目录或 notebooks/ 时都能 import gptoy
            ROOT = Path.cwd()
            if not (ROOT / "gptoy").is_dir() and (ROOT.parent / "gptoy").is_dir():
                ROOT = ROOT.parent
            if str(ROOT) not in sys.path:
                sys.path.insert(0, str(ROOT))

            import gptoy
            from gptoy.config import GPConfig
            from gptoy.data import koza1_target, make_koza1_data, make_target_data
            from gptoy.evolution import EvolutionEngine, run_gp
            from gptoy.experiments import (
                add_gaussian_noise,
                initialization_experiment,
                load_experiment,
                make_parsimony_fitness,
                make_train_val_test,
                run_budget_experiment,
                run_seed_batch,
                save_experiment,
                summarize_runs,
                tournament_experiment,
            )
            from gptoy.initialization import full as make_full, initial_population
            from gptoy.metrics import complexity, population_stats, structural_signature
            from gptoy.operators import evaluate, fitness, p_div
            from gptoy.representation import Node, str_to_tree
            from gptoy.selection import tournament_select
            from gptoy.tracing import EvolutionTracer, GenerationSnapshot, GeneticEvent
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
            from gptoy.viz import draw_tree, plot_tree

            plt.rcParams.update({"figure.dpi": 110, "axes.grid": True, "font.size": 9})

            print(f"Python      : {sys.version.split()[0]}")
            print(f"numpy       : {np.__version__}")
            print(f"matplotlib  : {matplotlib.__version__}")
            print(f"gptoy       : {gptoy.__file__}")
            print(f"工作目录    : {Path.cwd()}")


            def print_table(headers, rows):
                '''打印等宽文本表；按东亚字符显示宽度对齐（不引入额外依赖）。'''
                import unicodedata

                def width(text):
                    return sum(2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1
                               for ch in str(text))

                def pad(text, target):
                    text = str(text)
                    return text + " " * max(0, target - width(text))

                cells = [[str(h) for h in headers]] + [[str(v) for v in row] for row in rows]
                widths = [max(width(row[i]) for row in cells) for i in range(len(headers))]
                print("  ".join(pad(h, w) for h, w in zip(headers, widths)))
                print("  ".join("-" * w for w in widths))
                for row in rows:
                    print("  ".join(pad(v, w) for v, w in zip(row, widths)))
        """),
    ]
    return cells


# ---------------------------------------------------------------------------
# 第一篇
# ---------------------------------------------------------------------------
def chapter_0() -> list:
    return [
        md(r"""
            ## Chapter 0：GP 与符号回归

            **核心问题**：GP 在符号回归里到底搜索什么？训练/测试 MSE 很小，能说明公式被严格恢复了吗？

            ### 概念与数学定义

            **符号回归**：给定有限个观测 $(x_i, y_i)$，在一个由函数集与终止符集递归生成的
            表达式空间中寻找一个表达式 $T$，使 $T(x_i) \approx y_i$。与参数拟合不同，GP 同时搜索
            **结构**（用哪些运算、怎样嵌套）与**数值行为**。

            **搜索空间**：从终止符 $x$ 出发，反复用二元运算 $+,-,*,/$ 组合已有表达式，可得到全部
            有限表达式树。本项目的目标 $f(x)=x^4+x^3+x^2+x$ 位于该空间内，因为

            $$x^4+x^3+x^2+x = x(x+1)(x^2+1),$$

            而常数 $1$ 可以用保护除法 $x/x$ 合成。注意：**“目标在空间内”不等于“有限预算内一定搜到”**。

            **结构发现 vs 常数优化**：经典 GP 用随机常数或常数合成来近似系数；本项目的终止符只有
            $x$，系数只能靠 $x/x=1$、$x-x=0$ 等组合“合成”，这让常数优化本身也成为结构搜索的一部分。

            **Koza-1 完整定义**：

            - 训练集：$[-1,1]$ 上等距 20 点，标签由 $f(x)=x^4+x^3+x^2+x$ 生成；
            - 测试/泛化集：$[1,2]$ 上等距 10 点，标签同样由 $f$ 生成，但**从不参与搜索**；
            - 预算：种群 100，最多 50 代；判定阈值 $\mathrm{MSE} < 10^{-6}$。

            **三种“成功”必须分开**：

            1. **数值拟合**：训练/测试 MSE 小；
            2. **函数一致**：在关心的区间上逐点（或解析地）与目标一致；
            3. **结构恢复**：得到的树与某个等价推导结构相同。

            前两者可由有限样本上的数字支持，第三者需要结构判定或数学论证。本 Notebook 会反复展示
            “数值上完全一致、结构却不同”的例子。

            ### 具体小例子

            目标 $f(x)=x^4+x^3+x^2+x$ 可以写成因式分解形式 $x(x+1)(x^2+1)$。两种写法对应两棵
            结构不同的树（展开树 19 个节点、深度 6；因式树 15 个节点、深度 5），但在所有输入上
            函数值相同。**只看训练/测试 MSE，无法判断算法找到的是哪一种结构**——这正是“数值拟合”
            与“结构恢复”的分界线。

            ### 源码入口

            - 目标公式与数据：`gptoy/data.py` → `koza1_target`、`make_koza1_data`
            - 适应度：`gptoy/operators.py` → `fitness`
            - 主循环：`gptoy/evolution.py` → `run_gp`、`EvolutionEngine`

            > **我的预测（请填写）**：在 seed=3、种群 100、最多 50 代的预算下，你认为第 0 代的
            > best 训练 MSE 大概是什么量级？大约第几代能达标？最终表达式会不会与标准树
            > `(+ (+ (+ (* (* x x) (* x x)) (* x (* x x))) (* x x)) x)` 结构相同？为什么？
        """),
        code(r"""
            # 运行 Koza-1 基线：默认配置 + 固定 seed=3（与旧教学案例一致）
            SEED = 3
            cfg = GPConfig(seed=SEED)
            x_train, y_train, x_test, y_test = make_koza1_data(cfg)

            baseline = run_gp(cfg)

            print(f"训练集 {x_train.shape[0]} 点 @ [{cfg.train_lo}, {cfg.train_hi}]")
            print(f"测试集 {x_test.shape[0]} 点 @ [{cfg.test_lo}, {cfg.test_hi}]")
            print()
            print_table(
                ["指标", "数值"],
                [
                    ("第 0 代 best 训练 MSE", f"{baseline.gen0_mse:.6g}"),
                    ("首次达标代数", baseline.generation_found),
                    ("实际运行代数", baseline.n_generations),
                    ("训练 MSE", f"{baseline.train_mse:.3e}"),
                    ("测试 MSE", f"{baseline.test_mse:.3e}"),
                    ("训练适应度评估次数", baseline.evaluations),
                    ("运行耗时 (s)", f"{baseline.total_seconds:.2f}"),
                ],
            )
            print()
            print("最终表达式（S-表达式）：")
            print(" ", baseline.best_of_run.sexpr_str())
            print("最终表达式（中缀）：")
            print(" ", baseline.best_of_run.infix_str())
            print(f"结构规模：size={baseline.best_of_run.size()}, depth={baseline.best_of_run.depth()}")
        """),
        code(r"""
            # 数据长什么样：训练区间与泛化区间是分开的
            fig, axes = plt.subplots(1, 2, figsize=(11, 3.2))

            axes[0].plot(x_train, y_train, "o", ms=4, label="training points (20)")
            axes[0].plot(x_test, y_test, "s", ms=4, label="test points (10)")
            axes[0].set_xlabel("x")
            axes[0].set_ylabel("y")
            axes[0].set_title("Koza-1 data: train on [-1, 1], evaluate on [1, 2]")
            axes[0].legend()

            axes[1].semilogy(baseline.best_mse_history, "-o", ms=3, label="best-of-generation train MSE")
            axes[1].axhline(cfg.mse_target, color="tab:red", ls="--", lw=1, label="mse_target = 1e-6")
            axes[1].set_xlabel("generation")
            axes[1].set_ylabel("MSE (log)")
            axes[1].set_title("Convergence of the baseline run")
            axes[1].legend()
            fig.tight_layout()
        """),
        code(r"""
            # 数值一致 ≠ 结构相同：把基线结果与“标准树”比较
            canonical = str_to_tree(
                "(+ (+ (+ (* (* x x) (* x x)) (* x (* x x))) (* x x)) x)", cfg
            )
            grid = np.linspace(-2.0, 2.0, 401)
            gap = np.max(np.abs(evaluate(baseline.best_of_run, grid) - evaluate(canonical, grid)))

            print(f"结构与标准树相同？        {baseline.best_of_run == canonical}")
            print(f"标准树 size/depth：       {canonical.size()} / {canonical.depth()}")
            print(f"本次结果 size/depth：     {baseline.best_of_run.size()} / {baseline.best_of_run.depth()}")
            print(f"在 [-2, 2] 的 401 个点上最大绝对差：{gap:.3e}")
            print()
            print("阅读提示：请记录“结构是否相同”和“最大绝对差是多少”，并思考：")
            print("      这两个事实分别能支持什么、不能支持什么？请写到本章「观察与解释（请填写）」。")
        """),
        md(r"""
            ### 观察与解释（请填写）

            > **观察与解释（请填写）**：记录你实际看到的数字（gen0 MSE、首达代、最终表达式与结构），
            > 再回答：这次运行属于“数值拟合 / 函数一致 / 结构恢复”中的哪一级？你的证据是什么？
            > 测试集在这里能排除“只插值了训练点”吗？它还能排除什么、不能排除什么？

            **本章局限**：

            - 测试集只有 10 个点，且与训练集来自同一个已知目标函数；“测试 MSE 小”只说明在这 10 点上一致；
            - 基线运行只展示了一个种子，不能据此推断 GP 的平均成功率（Chapter 7 才讨论分布）；
            - 本次固定的 `seed=3` 来自仓库原有的教学种子搜索（`scripts/find_seed.py`），该搜索**同时使用
              训练与测试表现筛选**种子，属于教学用的成功样本，不能视为无偏的泛化成功率；
            - 本 Notebook 中所有“达标”只表示数值阈值，不表示严格的公式恢复。

            ### 本章笔记（请填写）

            **1. 核心概念（用自己的话重述）**

            > 待填写

            **2. 实现机制与源码位置**

            > 待填写

            **3. 我的实验问题与预测**

            > 待填写

            **4. 实验配置及观察结果**

            > 待填写

            **5. 我的理解、替代解释与证据边界**

            > 待填写

            **6. 本章总结与后续问题**

            > 待填写
        """),
    ]


def chapter_1() -> list:
    return [
        md(r"""
            ## Chapter 1：表达式树与程序表示

            **核心问题**：一个 GP 个体怎样被表示？不同的树什么时候表示同一个函数？保护运算改变了什么语义？

            ### 概念与数学定义

            **Function Set / Terminal Set**：本项目函数集为 $\{+,-,*,/\}$（全部二元），终止符集为
            $\{x\}$。一棵表达式树是一棵有序二叉树：内部节点是函数，叶子是终止符。

            **Genotype / Phenotype**：在经典树型 GP 中，基因型（被遗传操作修改的对象）与表现型
            （被求值的程序）都是同一棵树——这是树型 GP 与基因表达式编程（GEP）等表示的区别。

            **深度与规模**：

            - `depth(T)`：单节点树深度为 1，否则为 $1+\max_i \text{depth}(T_i)$；
            - `size(T)`：节点总数 $=1+\sum_i \text{size}(T_i)$；
            - 满二叉树深度 $d$ 的节点数为 $2^d-1$：深度增长会让规模指数增长，这是后面“膨胀”问题的根源。

            **语法 vs 语义**：S-表达式唯一确定结构（语法），但不同结构可以表示同一函数（语义）。
            例如标准树 `(+ (+ (+ (* (* x x) (* x x)) (* x (* x x))) (* x x)) x)` 与因式分解写法
            $x(x+1)(x^2+1)$ 的树结构完全不同，函数却处处相等。

            **保护运算（Protected Operations）**：`p_div(a, b)` 在 $b=0$ 时返回 $1.0$，而不是
            抛出异常或返回 NaN/Inf。这是 Koza 的经典约定，保证适应度可计算，但它**改变了数学语义**：

            $$\frac{x-x}{x}\Big|_{x=0} \;\text{按保护除法} = \frac{0}{0}\to 1.0,\quad
            \text{而“先约分得 }0\text{”只在 }x\neq 0\text{ 成立。}$$

            因此不能把普通代数化简当作保护除法下的全域等价变换；Chapter 10 会系统展示这一点。

            ### 具体小例子

            树 `(+ x (* x x))` 表示 $x+x^2$，深度 3、规模 5；在 $x=3$ 处值为 $3+9=12$。
            若把左边的 `x` 替换成 `(* x x)`，得到 $(x^2)+(x^2)=2x^2$——这正是 `replace_subtree` 做的事。

            ### 源码入口

            - 树结构、路径访问、解析与显示：`gptoy/representation.py` → `Node`、`Node.iter_nodes`、
              `Node.subtree_at`、`Node.replace_subtree`、`str_to_tree`
            - 递归求值与保护除法：`gptoy/operators.py` → `evaluate`、`p_div`
            - 树绘制：`gptoy/viz.py` → `draw_tree`、`plot_tree`
        """),
        md(r"""
            ### E01：构建、修改与比较表达式树

            **研究问题**：结构不同的两棵树能否表示同一函数？在保护除法下，哪些“显然”的代数变换会失效？

            **核心概念**：语法结构、函数语义、保护除法、结构相等 vs 数值相等。

            **可修改配置**（直接改下面单元里的变量后重跑）：

            - `TREE_A` / `TREE_B`：两棵要比较的 S-表达式；
            - `PROBE_POINTS`：手工检查的输入点。

            **运行入口**：下一代码单元（`# E01`）。

            **源码位置**：`gptoy/representation.py`、`gptoy/operators.py`、`gptoy/viz.py`。

            > **我的预测（请填写）**：先读下面两棵树的表达式，猜一猜 `TREE_A == TREE_B` 是 True 还是
            > False？它们的函数值在 $[-2,2]$ 上会差多少？`(x^2-1)/(x-1)` 在 $x=1$ 处会得到什么？
        """),
        code(r"""
            # E01 结构 vs 语义：不同结构，同一函数
            TREE_A = "(+ (+ (+ (* (* x x) (* x x)) (* x (* x x))) (* x x)) x)"  # 标准展开 x^4+x^3+x^2+x
            TREE_B = "(* (* x (+ x (/ x x))) (+ (* x x) (/ x x)))"             # 因式分解 x(x+1)(x^2+1)

            tree_a = str_to_tree(TREE_A, cfg)
            tree_b = str_to_tree(TREE_B, cfg)
            PROBE_POINTS = np.array([-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0])

            print("tree_a:", tree_a.sexpr_str())
            print("tree_b:", tree_b.sexpr_str())
            print_table(
                ["树", "size", "depth", "结构相等(a==b)", "hash 相等"],
                [
                    ("A", tree_a.size(), tree_a.depth(), tree_a == tree_b, hash(tree_a) == hash(tree_b)),
                    ("B", tree_b.size(), tree_b.depth(), "-", "-"),
                ],
            )
            print()
            print_table(
                ["x", "A(x)", "B(x)", "A(x)-B(x)"],
                [(f"{v:g}", f"{evaluate(tree_a, v):.12g}", f"{evaluate(tree_b, v):.12g}",
                  f"{evaluate(tree_a, v) - evaluate(tree_b, v):.3e}") for v in PROBE_POINTS],
            )
            dense = np.linspace(-2.0, 2.0, 401)
            print(f"\n更密网格上的最大绝对差：{np.max(np.abs(evaluate(tree_a, dense) - evaluate(tree_b, dense))):.3e}")
            print("阅读提示：比较两棵树的 size/depth 与逐点数值；结构不同是否等于函数不同？")
            print("      请把判断与理由写到本节「观察与解释（请填写）」。")
        """),
        code(r"""
            # E01 保护除法：形式代数变换在分母为 0 处失效
            t_ratio = str_to_tree("(/ (- (* x x) (/ x x)) (- x (/ x x)))", cfg)  # (x^2-1)/(x-1)
            t_zero_over_x = str_to_tree("(/ (- x x) x)", cfg)                    # (x-x)/x

            print("(x^2-1)/(x-1) 用保护除法求值，与 x+1 比较：")
            print_table(
                ["x", "保护除法结果", "x+1", "是否一致"],
                [(f"{v:g}", f"{evaluate(t_ratio, v):.6g}", f"{v + 1:.6g}",
                  "是" if abs(evaluate(t_ratio, v) - (v + 1)) < 1e-9 else "否")
                 for v in (-1.0, 0.5, 1.0, 2.0)],
            )
            print()
            print(f"(x-x)/x 在 x=0 处：{evaluate(t_zero_over_x, 0.0):.1f}（按“先约分”会预期 0）")
            print(f"x/x     在 x=0 处：{evaluate(str_to_tree('(/ x x)', cfg), 0.0):.1f}")
            print("原因：分母精确为 0 时 p_div 直接返回 1.0，不执行 a/b。")
        """),
        code(r"""
            # E01 修改树：replace_subtree 返回新树，原树保持不变
            modified = tree_b.replace_subtree((0,), str_to_tree("(* x x)", cfg))
            print("修改前 tree_b :", tree_b.sexpr_str())
            print("修改后 modified:", modified.sexpr_str())
            print("tree_b 是否被改动？", tree_b.sexpr_str() == TREE_B)

            fig, axes = plt.subplots(1, 2, figsize=(13, 3.4))
            draw_tree(axes[0], tree_b, title=f"tree_b (size={tree_b.size()}, depth={tree_b.depth()})")
            draw_tree(axes[1], modified, title="tree_b with subtree (0,) replaced by (* x x)")
            fig.tight_layout()
        """),
        md(r"""
            ### 观察与解释（请填写）

            > **观察与解释（请填写）**：两棵树的 size/depth 差多少？数值差是多少？这个差是 0 吗，
            > 还是只是浮点误差？`(x^2-1)/(x-1)` 在 $x=1$ 处偏离 $x+1$ 多少？
            > 由此说明：为什么“结构相同”与“函数相同”必须分开讨论？

            **本章局限**：这里的“函数相同”是在有限网格上数值验证的（`[-2,2]` 的 401 个点）；
            严格的函数相等需要数学论证。保护除法只在分母精确为 0 时替换，接近 0 的分母仍会放大误差；
            这些语义细节会直接影响“等价变换”是否成立。

            ### 本章笔记（请填写）

            **1. 核心概念（用自己的话重述）**

            > 待填写

            **2. 实现机制与源码位置**

            > 待填写

            **3. 我的实验问题与预测**

            > 待填写

            **4. 实验配置及观察结果**

            > 待填写

            **5. 我的理解、替代解释与证据边界**

            > 待填写

            **6. 本章总结与后续问题**

            > 待填写
        """),
    ]


def chapter_2() -> list:
    return [
        md(r"""
            ## Chapter 2：种群初始化

            **核心问题**：初始种群怎样生成？full / grow / ramped half-and-half 会带来怎样的搜索起点与偏差？

            ### 概念与数学定义

            设初始化深度参数为 $d$（本项目 `min_init_depth=2`、`max_init_depth=6`）。

            - **Full**：所有叶子都在第 $d$ 层，是一棵满二叉树，节点数固定为 $2^d-1$；
            - **Grow**：根强制为函数节点；其余节点从“函数集 ∪ 终止符集”中均匀抽取，
              抽到终止符就停止生长，因此深度 $\le d$ 且形状可变；
            - **Ramped half-and-half**：对每个深度桶 $d\in[2,6]$ 生成一批个体，桶内一半 full、一半 grow，
              最后打乱顺序。它同时提供“深度多样性”和“形状多样性”。

            **初始化偏差**：不同方法的深度分布、规模分布、结构唯一比例不同。Full 在固定深度上
            形状单一；Grow 的深浅偏置由「终止符在抽样池中的比例」决定——本项目是 4 个函数 + 1 个
            终止符，每个内部节点提前停止生长（抽到 x）的概率只有 $1/5$，所以 Grow 仍以接近最大深度
            为主，平均只比 Full 略浅；若终止符比例更高，Grow 才会明显偏向浅树。Ramped 覆盖多个
            深度桶。初始分布会影响后续选择和变异能接触到哪些结构，但**不能仅凭初始分布断言最终进化
            一定更好**——那需要对照实验（Chapter 8）。E02 会直接检验这三种方法在你机器上的实际分布。

            ### 具体小例子

            $d=5$ 时：full 树固定 31 个节点、深度 5；grow 树每个内部节点都在“继续生长”与“抽到 x
            停止”之间选择，由于只有 1 个终止符、4 个函数，多数树仍会长到接近深度 5，只有少数提前
            停止（例如根的两个子节点都抽到 x 时得到 3 个节点的浅树）。用同一随机种子生成同一个
            种群，可以逐位复现；E02 的深度直方图会让你看到这一偏置的实际大小。

            ### 源码入口

            - `gptoy/initialization.py` → `full`、`grow`、`initial_population`、`INIT_METHODS`
            - 统计指标：`gptoy/metrics.py` → `population_stats`（深度/规模直方图、结构唯一比例）
        """),
        md(r"""
            ### E02：三种初始化策略的搜索起点

            **研究问题**：在相同种群规模与深度参数下，full / grow / ramped 生成的种群在深度分布、
            规模分布与结构多样性上有什么差异？这些差异意味着怎样的搜索起点？

            **核心概念**：初始化分布、深度桶、结构唯一比例。

            **可修改配置**（改下面单元顶部的 `E02_DEPTH`、`E02_POP_SIZE`、`E02_METHODS` 后重跑）：

            - `E02_DEPTH`：初始化深度上限；
            - `E02_POP_SIZE`：种群规模；
            - `E02_METHODS`：要比较的方法元组。

            **运行入口**：下一代码单元（`# E02`）。

            **源码位置**：`gptoy/initialization.py`、`gptoy/metrics.py`、`gptoy/experiments.py`
            → `initialization_experiment`。

            > **我的预测（请填写）**：三种方法的平均深度、最大深度、平均节点数、结构唯一比例
            > 大概会怎样排序？full 的深度直方图会是什么形状？ramped 呢？
        """),
        code(r"""
            # E02 三种初始化方法：深度/规模分布与结构唯一性
            E02_DEPTH = 5
            E02_POP_SIZE = 60
            E02_METHODS = ("full", "grow", "ramped")

            init_rows = initialization_experiment(
                cfg, E02_METHODS, depth=E02_DEPTH, population_size=E02_POP_SIZE, seed=0
            )
            print_table(
                ["method", "mean_depth", "max_depth", "mean_size", "max_size", "unique", "unique_ratio"],
                [
                    (row["method"], f"{row['mean_depth']:.2f}", row["max_depth"],
                     f"{row['mean_size']:.2f}", row["max_size"], row["unique_structures"],
                     f"{row['unique_ratio']:.3f}")
                    for row in init_rows
                ],
            )
            print()
            for row in init_rows:
                print(f"{row['method']:>7} 深度直方图: {row['depth_histogram']}")
                print(f"{'':>7} 规模直方图: {dict(sorted(row['size_histogram'].items()))}")
        """),
        code(r"""
            # E02 可视化：深度分布 + 规模/唯一性对比
            fig, axes = plt.subplots(1, 2, figsize=(12, 3.4))

            all_depths = sorted({d for row in init_rows for d in row["depth_histogram"]})
            width = 0.8 / len(init_rows)
            for i, row in enumerate(init_rows):
                heights = [row["depth_histogram"].get(d, 0) for d in all_depths]
                axes[0].bar([d + (i - (len(init_rows) - 1) / 2) * width for d in all_depths],
                            heights, width=width, label=row["method"])
            axes[0].set_xlabel("tree depth")
            axes[0].set_ylabel("count")
            axes[0].set_title(f"Depth distribution of initial populations (depth<={E02_DEPTH})")
            axes[0].set_xticks(all_depths)
            axes[0].legend()

            xpos = np.arange(len(init_rows))
            axes[1].bar(xpos - 0.2, [row["mean_size"] for row in init_rows], width=0.4, label="mean size")
            axes[1].bar(xpos + 0.2, [row["max_size"] for row in init_rows], width=0.4, label="max size")
            axes[1].set_xticks(xpos)
            axes[1].set_xticklabels([row["method"] for row in init_rows])
            axes[1].set_ylabel("nodes")
            axes[1].set_title("Size: mean vs max")
            axes[1].legend()

            fig.tight_layout()
            print("唯一比例：", {row["method"]: round(row["unique_ratio"], 3) for row in init_rows})
        """),
        md(r"""
            ### 观察与解释（请填写）

            > **观察与解释（请填写）**：三种方法的平均深度和规模差多少？哪一种的深度分布最集中？
            > ramped 是否覆盖了多个深度桶？结构唯一比例说明什么？如果只改变深度上限或种群规模，
            > 这些结论会变吗？请记录你实际改参数后的结果。

            **本章局限**：这里只观察初始化后的**起点**，没有让它们各自进化。初始化分布与最终搜索
            成败之间没有必然的单向关系；要判断哪种起点更好，必须进入同一预算下的对照实验。

            ### 本章笔记（请填写）

            **1. 核心概念（用自己的话重述）**

            > 待填写

            **2. 实现机制与源码位置**

            > 待填写

            **3. 我的实验问题与预测**

            > 待填写

            **4. 实验配置及观察结果**

            > 待填写

            **5. 我的理解、替代解释与证据边界**

            > 待填写

            **6. 本章总结与后续问题**

            > 待填写
        """),
    ]


def chapter_3() -> list:
    return [
        md(r"""
            ## Chapter 3：适应度评价

            **核心问题**：适应度怎样定义？MSE 能评价什么、不能评价什么？训练/验证/测试各自负责什么？

            ### 概念与数学定义

            **经验风险（MSE）**：

            $$\mathrm{MSE}(T; D)=\frac{1}{|D|}\sum_{(x,y)\in D}\big(T(x)-y\big)^2 .$$

            本项目用训练集 MSE 作为唯一搜索信号：`fit = fitness(tree, x_train, y_train)`。
            `fitness` 只接收 `(tree, x, y)`，内部不出现目标公式；这是“从零实现 GP”的硬约束。

            **复杂度**：`size(T)` 与 `depth(T)`。MSE 与复杂度是两个不同维度：同样小的 MSE 可以来自
            很小或很大的树；同样复杂度的树可以有很不同的误差。

            **三个数据集的职责**：

            | 数据集 | 用途 | 能否反馈到搜索 |
            |---|---|---|
            | 训练集 | 计算适应度、驱动选择 | 能，唯一允许的搜索信号 |
            | 验证集 | 调参/模型选择（如选择 penalty、判断过拟合） | 只能用于事先声明的选择规则 |
            | 测试集 | 最终评价、报告泛化 | **不能**；不能用来挑种子或调参 |

            **为什么 MSE 不足以评价“公式恢复”**：

            1. **欠定**：训练点太少时，多个不同结构可以有相同（甚至 0）训练误差；
            2. **等价类**：同一函数有无限多种表达式结构；
            3. **外推**：训练区间上一致，不代表区间外一致；
            4. **有限样本噪声**：小 MSE 可能只是拟合了噪声。

            ### 具体小例子

            取目标 $f(x)=x^2$，训练点 $x\in\{-1,0,1\}$。$T_1=x^2$、$T_2=x^4$、$T_3=x^6$ 在
            这三个点上都给出 $(1,0,1)$，训练 MSE 全为 0；但在 $x=2$ 处分别给出 $4,16,64$。
            **训练误差完全相同，三个表达式却是不同的函数。**

            ### 源码入口

            - 适应度与求值：`gptoy/operators.py` → `fitness`、`evaluate`
            - 复杂度：`gptoy/metrics.py` → `complexity`
            - 数据：`gptoy/data.py` → `make_koza1_data`、`make_target_data`
        """),
        md(r"""
            ### E03：MSE 能区分什么、不能区分什么

            **研究问题**：结构不同但数学等价的表达式，MSE 是否相同？训练误差相同（甚至为 0）的
            候选，在独立测试点上的表现是否相同？

            **核心概念**：经验风险、等价类、欠定问题、外推、训练/测试职责。

            **可修改配置**（改下面单元顶部的 `KOZA_CANDIDATES` / `SMALL_TRAIN_X` 后重跑）：

            - `KOZA_CANDIDATES`：在 Koza-1 训练/测试集上比较的若干候选结构；
            - `SMALL_TRAIN_X`：欠定实验的训练点（默认三个点）。

            **运行入口**：接下来两个代码单元（`# E03`）。

            **源码位置**：`gptoy/operators.py` → `fitness`；`gptoy/data.py`。

            > **我的预测（请填写）**：三个候选里，哪些训练 MSE 接近 0？它们的测试 MSE 会一样吗？
            > 在三点训练集上，$x^2$、$x^4$、$x^6$ 的训练/测试 MSE 各是多少？
        """),
        code(r"""
            # E03-A 在 Koza-1 数据上比较三个结构
            KOZA_CANDIDATES = {
                "exact": "(+ (+ (+ (* (* x x) (* x x)) (* x (* x x))) (* x x)) x)",
                "factored": "(* (* x (+ x (/ x x))) (+ (* x x) (/ x x)))",
                "partial_missing_x": "(+ (+ (* (* x x) (* x x)) (* x (* x x))) (* x x))",
            }
            print_table(
                ["候选", "训练 MSE", "测试 MSE", "size", "depth"],
                [
                    (name,
                     f"{fitness(str_to_tree(s, cfg), x_train, y_train):.6g}",
                     f"{fitness(str_to_tree(s, cfg), x_test, y_test):.6g}",
                     str_to_tree(s, cfg).size(), str_to_tree(s, cfg).depth())
                    for name, s in KOZA_CANDIDATES.items()
                ],
            )
            print()
            print("阅读提示：对比 exact / factored / partial_missing_x 的训练 MSE、测试 MSE 与结构。")
            print("      仅凭这些数字，你能区分“数值拟合”与“结构恢复”吗？请写下你的依据。")
        """),
        code(r"""
            # E03-B 欠定训练集：三个不同结构有相同的 0 训练误差，测试表现却不同
            SMALL_TRAIN_X = np.array([-1.0, 0.0, 1.0])
            SMALL_TRAIN_Y = SMALL_TRAIN_X ** 2
            PROBE_X = np.array([-2.0, -1.5, -0.5, 0.5, 1.5, 2.0])

            hypotheses = {"x^2": "(* x x)", "x^4": "(* (* x x) (* x x))",
                          "x^6": "(* (* x x) (* (* x x) (* x x)))"}
            print_table(
                ["候选", "训练 MSE（3 点）", "独立测试 MSE", "size"],
                [
                    (name,
                     f"{np.mean((evaluate(str_to_tree(s, cfg), SMALL_TRAIN_X) - SMALL_TRAIN_Y) ** 2):.3g}",
                     f"{np.mean((evaluate(str_to_tree(s, cfg), PROBE_X) - PROBE_X ** 2) ** 2):.3g}",
                     str_to_tree(s, cfg).size())
                    for name, s in hypotheses.items()
                ],
            )

            fig, ax = plt.subplots(figsize=(7, 3.4))
            xs = np.linspace(-2.2, 2.2, 300)
            for name, s in hypotheses.items():
                ax.plot(xs, evaluate(str_to_tree(s, cfg), xs), label=f"{name} hypothesis")
            ax.plot(xs, xs ** 2, "k--", lw=1, label="true target x^2")
            ax.plot(SMALL_TRAIN_X, SMALL_TRAIN_Y, "ko", ms=6, label="training points (3)")
            ax.plot(PROBE_X, PROBE_X ** 2, "k^", ms=5, mfc="none", label="independent test points")
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_title("Three hypotheses fit the 3 training points exactly (MSE=0)")
            ax.legend(fontsize=7)
            fig.tight_layout()
        """),
        md(r"""
            ### 观察与解释（请填写）

            > **观察与解释（请填写）**：三个假设的训练 MSE 是否完全相同？测试 MSE 相差多少量级？
            > 如果你只知道训练 MSE，能否从中挑出“正确”的 $x^2$？需要什么额外信息才能做到
            > （更多训练点？区间先验？结构复杂度偏好？独立验证集？）请写下你的推理与证据边界。

            **本章局限与诚实性说明**：

            - 三点训练集是**刻意构造的说明性例子**，用来展示欠定问题，不代表典型 GP 运行；
            - 独立测试点也来自同一个 $x^2$ 目标，但位于训练区间之外，因此度量的是外推；
            - 在真实问题中，“测试 MSE 小”依然不能单独证明结构恢复，只能提供有限样本上的证据。

            ### 本章笔记（请填写）

            **1. 核心概念（用自己的话重述）**

            > 待填写

            **2. 实现机制与源码位置**

            > 待填写

            **3. 我的实验问题与预测**

            > 待填写

            **4. 实验配置及观察结果**

            > 待填写

            **5. 我的理解、替代解释与证据边界**

            > 待填写

            **6. 本章总结与后续问题**

            > 待填写
        """),
    ]
