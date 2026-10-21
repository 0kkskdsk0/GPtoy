"""GPtoy 学习实验室内容（下半部分）：Chapter 4–11 与附录。"""

from lab_content import code, md

CHAPTERS_B = [
    {
        "num": 4,
        "title": "选择机制",
        "part": 0,
        "core_question": "选择压力从哪里来？锦标赛规模 k 改变了什么？“选择更强”是否等于“最终进化更好”？",
        "source": "`gptoy/selection.py`（`tournament_select`、`tournament_select_indices`）、`gptoy/experiments.py`（`tournament_experiment`）、`gptoy/metrics.py`（`population_stats`）",
        "experiments": ["E04"],
        "concepts": [
            "选择压力：更优个体被选为父本的概率优势；锦标赛规模 k 是调节旋钮",
            "锦标赛选择有放回：每轮独立抽 k 个个体，MSE 最小者胜；同一个亲本可以被多次选中",
            "若种群规模 N、只有唯一最优个体，则它在一次锦标赛中被选中的概率为 $1-(1-1/N)^k$",
            "Exploration vs Exploitation：k 小接近随机抽样，k 大迅速集中到少数个体",
            "选择压力影响父本的适应度分布与结构多样性，但不保证最终进化结果更好",
        ],
    },
    {
        "num": 5,
        "title": "遗传操作",
        "part": 0,
        "core_question": "交叉、变异、复制分别怎样改变结构？深度限制与回退机制在保护什么？",
        "source": "`gptoy/variation.py`（`crossover`、`crossover_traced`、`mutate`、`mutate_traced`、`reproduce`、`apply_variation`、`VariationRecord`）、`gptoy/representation.py`（`subtree_at`、`replace_subtree`）",
        "experiments": ["E05"],
        "concepts": [
            "子树交叉：两个父本各选一个交叉点，交换子树，产生两个子代",
            "子树变异：选一个节点，用一棵新的 grow 子树替换；新子树深度受剩余深度预算约束",
            "复制：父本深拷贝；精英保留可看作特殊的复制",
            "继承与破坏：交叉可能保留有用子结构，也可能拆散它；变异可引入新结构，也可只是中性替换",
            "深度上限与回退：交叉后超过 max_tree_depth 则重试选点，重试耗尽后回退为父本副本",
        ],
    },
    {
        "num": 6,
        "title": "完整进化过程",
        "part": 0,
        "core_question": "一代完整进化包含哪些步骤？逐代执行与一次性运行是否等价？停止条件怎样影响观察？",
        "source": "`gptoy/evolution.py`（`EvolutionEngine`、`initialize`、`step`、`run`、`run_gp`、`RunResult`）、`gptoy/tracing.py`（`GenerationSnapshot`、`GeneticEvent`、`EvolutionTracer`）",
        "experiments": ["E06"],
        "concepts": [
            "Generational：整代同步替换；精英保留把上一代最优个体直接复制到下一代",
            "best-of-generation 与 best-of-run：前者是每代最优，后者是历代最优；Hall of Fame 记录历代变优的个体",
            "停止条件：训练 MSE 达到阈值（可关闭）或达到 max_generations",
            "收敛曲线：精英保留使 best MSE 单调不增（同一适应度定义下），台阶表示发现更好个体",
            "逐代 step() 与 run_gp() 必须使用同一引擎、同一 RNG 序列，历史才能逐位一致",
        ],
    },
    {
        "num": 7,
        "title": "随机性与搜索预算",
        "part": 1,
        "core_question": "GP 的随机性来自哪里？预算怎样限制搜索？不同随机种子的结果分布说明了什么？",
        "source": "`gptoy/evolution.py`（`evaluations`、`stop_reason`）、`gptoy/experiments.py`（`run_seed_batch`、`run_budget_experiment`、`summarize_runs`）、`gptoy/utils.py`（`make_rng`）",
        "experiments": ["E07"],
        "concepts": [
            "随机性来源：初始化、父本抽样、交叉点/变异点、grow 子树形状；本项目全部来自单一 random.Random",
            "搜索预算：训练适应度评估次数 evaluations，是比墙钟时间更稳定的预算单位",
            "首达代：训练 MSE 第一次低于阈值的代数；没有达标则为 None",
            "重复性：同 seed、同配置逐位可复现；换 seed 得到的是同一随机过程的不同样本",
            "多种子统计：应报告全部种子与失败案例，区分训练达标率、测试达标率与结构恢复代理率",
        ],
    },
]


# ---------------------------------------------------------------------------
# Chapter 4
# ---------------------------------------------------------------------------
def chapter_4() -> list:
    return [
        md(r"""
            ## Chapter 4：选择机制

            **核心问题**：选择压力从哪里来？锦标赛规模 k 改变了什么？“选择更强”是否等于“最终进化更好”？

            ### 概念与数学定义

            **适应度比例选择**会按 $p_i \propto 1/\mathrm{MSE}_i$ 抽样，但要求适应度非负、对异常值敏感。
            本项目使用**锦标赛选择**：每轮有放回地随机抽取 $k$ 个个体，取 $\mathrm{MSE}$ 最小者作为父本；
            重复直到凑够所需父本数。

            设种群规模为 $N$，且只有一个全局最优个体。一次锦标赛没有抽到它的概率是 $(1-1/N)^k$，
            所以它被选中的概率是

            $$P_{\text{best}}(k)=1-\left(1-\frac{1}{N}\right)^k .$$

            $k=1$ 时 $P=1/N$（纯随机抽样）；$k$ 增大时 $P$ 迅速接近 1。**选择压力**就是这种“更优者
            更容易成为父本”的倾向。

            **有放回**意味着同一个亲本可以多次被选中，种群多样性可能迅速下降；但“父本选择压力更强”
            只是说**一步选择**更偏向当前最优，并不保证多代之后最终结果更好——后者还取决于变异算子、
            精英保留、停止条件与搜索空间结构。

            ### 具体小例子

            $N=5$、$k=1,2,5$ 时，若最优个体唯一，则 $P_{\text{best}}=0.2,0.36,0.672$。
            注意 $k=N$ 也不是 1：有放回抽样仍可能漏掉最优个体（概率 $(4/5)^5\approx0.33$）。

            ### 源码入口

            - `gptoy/selection.py` → `tournament_select`、`tournament_select_indices`
            - `gptoy/experiments.py` → `tournament_experiment`
            - `gptoy/metrics.py` → `population_stats`（父本适应度分布与结构唯一性）
        """),
        md(r"""
            ### E04：固定种群，只改变锦标赛规模

            **研究问题**：在初始种群与适应度完全固定时，只改变 $k\in\{1,2,7,12\}$，被选中父本的
            适应度分布与结构多样性如何变化？

            **核心概念**：选择压力、有放回抽样、exploration/exploitation、结构多样性。

            **可修改配置**（改下面单元顶部变量后重跑）：

            - `E04_TOURNAMENT_SIZES`：要比较的 k；
            - `E04_N_SELECT`：每个 k 选多少个父本（样本量越大，分布估计越稳定）；
            - `E04_POP_SIZE`：固定种群规模。

            **运行入口**：下一代码单元（`# E04`）。

            **源码位置**：`tournament_experiment` 内部调用 `tournament_select_indices`；
            每一行还返回选中下标，便于你复算。

            > **我的预测（请填写）**：k 从 1 增大到 12 时，父本 MSE 的中位数会怎样变化？
            > 被选中父本的结构唯一数会怎样变化？父本 MSE 的**均值**是否适合作为比较指标？为什么？
        """),
        code(r"""
            # E04 固定初始种群与适应度，只改变 tournament size
            E04_TOURNAMENT_SIZES = (1, 2, 7, 12)
            E04_N_SELECT = 40
            E04_POP_SIZE = 60

            e04_cfg = replace(cfg, population_size=E04_POP_SIZE, min_init_depth=2, max_init_depth=5)
            e04_population = initial_population(e04_cfg, make_rng(7), method="ramped")
            e04_fitnesses = [fitness(tree, x_train, y_train) for tree in e04_population]

            e04_rows = tournament_experiment(
                e04_cfg, e04_population, e04_fitnesses,
                E04_TOURNAMENT_SIZES, n_select=E04_N_SELECT, seed=11,
            )

            print(f"固定种群：{E04_POP_SIZE} 个个体；种群 MSE 均值={np.mean(e04_fitnesses):.4g}，"
                  f"中位数={np.median(e04_fitnesses):.4g}，最大={np.max(e04_fitnesses):.4g}")
            print("（MSE 分布是重尾的：均值会被少数极差个体拉高，因此下面同时给出中位数）")
            print()
            print_table(
                ["k", "父本 MSE 均值", "中位数", "最小", "最大", "唯一结构数", "唯一比例", "预测 P_best"],
                [
                    (row["tournament_size"], f"{row['parent_fitness_mean']:.4g}",
                     f"{row['parent_fitness_median']:.4g}", f"{row['parent_fitness_min']:.4g}",
                     f"{row['parent_fitness_max']:.4g}", row["unique_parent_structures"],
                     f"{row['unique_parent_ratio']:.3f}",
                     f"{1 - (1 - 1 / E04_POP_SIZE) ** row['tournament_size']:.3f}")
                    for row in e04_rows
                ],
            )
        """),
        code(r"""
            # E04 可视化：父本适应度分布（对数尺度）与结构多样性
            fig, axes = plt.subplots(1, 2, figsize=(12, 3.6))

            log_fits = [np.log10(np.asarray(row["parent_fitnesses"], dtype=float) + 1e-12) for row in e04_rows]
            axes[0].boxplot(log_fits)
            axes[0].set_xticks(range(1, len(e04_rows) + 1))
            axes[0].set_xticklabels([f"k={row['tournament_size']}" for row in e04_rows])
            axes[0].axhline(np.log10(np.median(e04_fitnesses) + 1e-12), color="tab:red",
                            ls="--", lw=1, label="population median")
            axes[0].set_ylabel("log10(parent MSE + 1e-12)")
            axes[0].set_title(f"Parent fitness distribution (n_select={E04_N_SELECT})")
            axes[0].legend(fontsize=7)

            axes[1].bar(range(len(e04_rows)), [row["unique_parent_ratio"] for row in e04_rows],
                        color="tab:green")
            axes[1].set_xticks(range(len(e04_rows)))
            axes[1].set_xticklabels([f"k={row['tournament_size']}" for row in e04_rows])
            axes[1].set_ylabel("unique selected structures / n_select")
            axes[1].set_ylim(0, 1.05)
            axes[1].set_title("Structural diversity of selected parents")
            fig.tight_layout()

            print("提示：本图只回答“这一步选择偏向谁”。它不能回答“多代之后谁进化得更好”。")
        """),
        md(r"""
            ### 观察与解释（请填写）

            > **观察与解释（请填写）**：k 增大时父本 MSE 的中位数怎样变化？唯一结构比例怎样变化？
            > 理论 $P_{\text{best}}$ 与实际观察一致吗？为什么 k=1 的均值与种群均值可能差很多？
            > “选择压力更强”能推出“最终进化一定更好”吗？请写下你需要什么实验才能检验后半句。

            **本章局限**：只观察了一步选择，没有进化；父本适应度分布受初始种群随机性影响，换一个初始化
            种子结论可能变化（Chapter 7 讨论种子分布）。重尾分布下均值不稳定，中位数更稳健，但也不能
            代表整个分布。

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


# ---------------------------------------------------------------------------
# Chapter 5
# ---------------------------------------------------------------------------
def chapter_5() -> list:
    return [
        md(r"""
            ## Chapter 5：遗传操作

            **核心问题**：交叉、变异、复制分别怎样改变结构？深度限制与回退机制在保护什么？

            ### 概念与数学定义

            本项目实现三种算子（函数集全为二元运算，因此树的每个内部节点有两个子节点）：

            | 算子 | 默认概率 | 做法 | 深度控制 |
            |---|---|---|---|
            | `crossover(p1, p2)` | 0.7 | 两父本各均匀选一个节点，交换其子树，产生两个子代 | 子代深度 $>17$ 时重选交叉点，最多 20 次；仍超限则回退为父本副本 |
            | `mutate(ind)` | 0.2 | 均匀选一个节点，用 grow 新子树替换 | 新子树深度 $\le \max\_depth - $ 变异点路径长度 |
            | `reproduce(ind)` | 0.1 | 深拷贝 | 无 |

            记交叉点路径长度为 $L_1,L_2$，被交换子树深度为 $D_1,D_2$，则两个子代深度分别为
            $L_1+D_2$、$L_2+D_1$；只有两者都 $\le \max\_depth$ 时这次交叉才被接受。

            **继承与破坏**：交叉可以保留父本的有用子结构（如果交换点选在无关子树），也可以拆散它；
            变异可能引入新结构，也可能只是**中性替换**（结构变了、适应度几乎不变）。

            **复杂度增长**：交叉/变异都可能让树变大；在只有 MSE 的目标下，稍大的树偶尔能拟合得更好，
            于是“膨胀”（bloat）可能被间接奖励——这需要实验检验，不能预设（Chapter 8）。

            ### 具体小例子

            `p1 = (+ x (* x x))`（$x+x^2$）、`p2 = (* (+ x x) x)`（$2x^2$）。若在 p1 的左叶子与
            p2 的右子树上交叉，子代可能变成 `(+ (* x x) (* x x))`（$2x^2$）等结构。

            ### 源码入口

            - `gptoy/variation.py` → `crossover`、`mutate`、`reproduce`、`apply_variation`
            - 追踪版本：`crossover_traced`、`mutate_traced`、`apply_variation_traced`（返回
              `CrossoverTrace` / `MutationTrace` / `VariationRecord`，RNG 消费顺序与普通版本完全一致）
            - 树路径操作：`gptoy/representation.py` → `iter_nodes`、`subtree_at`、`replace_subtree`
        """),
        md(r"""
            ### E05：真实执行一次交叉与一次变异

            **研究问题**：一次交叉/变异究竟改动了哪棵子树？子代结构与适应度相对父本如何变化？
            深度上限触发回退时，算子实际返回什么？

            **核心概念**：交叉点/变异点、继承/破坏、中性变化、深度预算与回退。

            **可修改配置**（改下面单元顶部变量后重跑）：

            - `E05_PARENT_INDICES`：从固定种群中选哪两个父本；
            - `E05_DEMO_MAX_DEPTH`：演示用深度上限（**调小以便画图**；默认算法仍用 17）；
            - `E05_CROSSOVER_SEED` / `E05_MUTATION_SEED`：算子随机种子。

            **运行入口**：接下来两个代码单元（`# E05`）。

            **源码位置**：`gptoy/variation.py` 的 traced 接口；`gptoy/viz.py` → `draw_tree`。

            > **我的预测（请填写）**：交叉点被标红后，子代是否真的替换了那棵子树？子代适应度会更好、
            > 更差还是持平？变异点是叶子还是内部节点？变异后结构一定改变吗？回退时子代与父本是什么关系？
        """),
        code(r"""
            # E05-A 真实交叉：记录交叉点、子代、前后适应度与结构变化
            E05_PARENT_INDICES = (0, 1)
            E05_DEMO_MAX_DEPTH = 6          # 仅用于让图可读；算法默认 max_tree_depth=17
            E05_CROSSOVER_SEED = 21

            e05_cfg = replace(e04_cfg, max_tree_depth=E05_DEMO_MAX_DEPTH)
            p1 = e04_population[E05_PARENT_INDICES[0]]
            p2 = e04_population[E05_PARENT_INDICES[1]]
            cross = crossover_traced(p1, p2, e05_cfg, make_rng(E05_CROSSOVER_SEED))

            print("父本 1:", p1.sexpr_str())
            print("父本 2:", p2.sexpr_str())
            print(f"交叉点路径：p1={cross.point1}, p2={cross.point2}；fallback={cross.fallback}")
            print()
            print("子代 1:", cross.child1.sexpr_str())
            print("子代 2:", cross.child2.sexpr_str())
            print()
            print_table(
                ["个体", "size", "depth", "训练 MSE", "相对父本1", "相对父本2"],
                [
                    (name, tree.size(), tree.depth(), f"{fitness(tree, x_train, y_train):.6g}",
                     "-", "-")
                    for name, tree in (("parent1", p1), ("parent2", p2))
                ]
                + [
                    (name, tree.size(), tree.depth(), f"{fitness(tree, x_train, y_train):.6g}",
                     "更好" if fitness(tree, x_train, y_train) < fitness(p1, x_train, y_train) else "不更好",
                     "更好" if fitness(tree, x_train, y_train) < fitness(p2, x_train, y_train) else "不更好")
                    for name, tree in (("child1", cross.child1), ("child2", cross.child2))
                ],
            )
            print(f"交叉是否改变结构？结构确实变化：{cross.child1 != p1 or cross.child2 != p2}")
        """),
        code(r"""
            # E05-A 绘图：高亮父本中的交叉点
            fig, axes = plt.subplots(1, 3, figsize=(15, 3.6))
            draw_tree(axes[0], p1, title="parent1 (highlight = crossover point)", highlight_paths=[cross.point1])
            draw_tree(axes[1], p2, title="parent2 (highlight = crossover point)", highlight_paths=[cross.point2])
            draw_tree(axes[2], cross.child1, title="child1")
            fig.tight_layout()
        """),
        code(r"""
            # E05-B 真实变异 + 深度回退演示
            E05_MUTATION_SEED = 22
            mutation = mutate_traced(p1, e05_cfg, make_rng(E05_MUTATION_SEED))

            print(f"变异点路径：{mutation.point}（长度 {len(mutation.point)}）；新子树深度 {mutation.new_subtree_depth}")
            print("变异前:", p1.sexpr_str())
            print("变异后:", mutation.child.sexpr_str())
            print_table(
                ["个体", "size", "depth", "训练 MSE", "结构是否改变"],
                [
                    ("变异前", p1.size(), p1.depth(), f"{fitness(p1, x_train, y_train):.6g}", "-"),
                    ("变异后", mutation.child.size(), mutation.child.depth(),
                     f"{fitness(mutation.child, x_train, y_train):.6g}",
                     mutation.child != p1),
                ],
            )
            print()

            # 深度回退：两棵深度 10 的满树在 max_tree_depth=5 下交叉，必然超深 -> 回退为父本副本
            deep1 = make_full(10, cfg, make_rng(1))
            deep2 = make_full(10, cfg, make_rng(2))
            fallback_cfg = replace(cfg, max_tree_depth=5)
            fallback_trace = crossover_traced(deep1, deep2, fallback_cfg, make_rng(3))
            print(f"回退演示：父本深度均为 {deep1.depth()}，max_tree_depth=5")
            print(f"  fallback={fallback_trace.fallback}，child1==parent1：{fallback_trace.child1 == deep1}，"
                  f"child2==parent2：{fallback_trace.child2 == deep2}")

            fig, axes = plt.subplots(1, 2, figsize=(12, 3.6))
            draw_tree(axes[0], p1, title="before mutation (highlight = mutation point)",
                      highlight_paths=[mutation.point])
            draw_tree(axes[1], mutation.child, title="after mutation")
            fig.tight_layout()
        """),
        md(r"""
            ### 观察与解释（请填写）

            > **观察与解释（请填写）**：交叉点是否落在你从表达式上预期的位置？子代是否真的交换了那棵
            > 子树？子代适应度比父本更好还是更差？变异点是叶子还是内部节点，结构改变后适应度变了多少？
            > 回退时子代与父本是什么关系，为什么必须回退？请把“结构变化”与“适应度变化”分开描述。

            **本章局限**：演示用的深度上限被调小（6），因此画出的树比默认算法中的树小；这只影响
            可视化，不影响算子语义。单次算子执行结果是随机样本，不能据此推断算子的平均效果。

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


# ---------------------------------------------------------------------------
# Chapter 6
# ---------------------------------------------------------------------------
def chapter_6() -> list:
    return [
        md(r"""
            ## Chapter 6：完整进化过程

            **核心问题**：一代完整进化包含哪些步骤？逐代执行与一次性运行是否等价？停止条件怎样影响观察？

            ### 概念与数学定义

            **生成式进化（generational）**：每一代用当前种群产生等规模的新种群，整代同步替换。
            本项目的新一代构成为

            $$\text{new} = \underbrace{[\,\text{elite}\,]\times e}_{\text{精英保留}} \;\Vert\;
            \texttt{apply\_variation}(\text{parents}),$$

            其中 $e=\text{elitism\_size}=1$，父本由锦标赛从上一代选出 $N-e$ 个。

            **best-of-generation / best-of-run / Hall of Fame**：

            - best-of-generation：当前代适应度最小的个体；
            - best-of-run：截至当前所有代中最好的个体（按适应度严格变优时更新）；
            - Hall of Fame：历代 best-of-run 的副本序列，便于回看“何时发现了更好的解”。

            **停止条件**：`stop_on_target=True`（默认）时训练 MSE 首次低于 `mse_target` 就停止；
            `stop_on_target=False` 时继续跑满 `max_generations`，用于观察长期行为（Chapter 8）。

            **收敛曲线**：由于精英保留，下一代一定包含上一代最优个体，所以 **best MSE 单调不增**
            （在同一个适应度定义下）。曲线的台阶对应“发现了更好的个体”。

            ### 具体小例子

            若第 3 代最优个体被复制到第 4 代，而第 4 代没有更好的子代，则第 4 代 best MSE 与第 3 代
            相同；一旦某个子代更好，best MSE 下降。若第 0 代就达标，则 `generation_found=0`——
            这正是旧版种子搜索要避免的情况（否则看不到进化过程）。

            ### 源码入口

            - 引擎与结果：`gptoy/evolution.py` → `EvolutionEngine.initialize/step/run`、`run_gp`、`RunResult`
            - 快照与事件：`gptoy/tracing.py` → `GenerationSnapshot`、`GeneticEvent`、`EvolutionTracer`
            - 评估计数：`EvolutionEngine.evaluations`（每评价一代种群计 `population_size` 次）
        """),
        md(r"""
            ### E06：逐代执行一次完整的 GP

            **研究问题**：如何显式地初始化、逐代推进、查看每代最优与真实遗传事件，并连续运行到停止？
            逐代执行与一次性 `run_gp` 的历史是否完全一致？

            **核心概念**：generational、elitism、best-of-run、Hall of Fame、停止条件、收敛曲线、
            逐代/批量同引擎。

            **可修改配置**（改下面单元顶部变量后重跑）：

            - `E06_SEED`、`E06_STEPS_BEFORE_PRINT`：种子与先手动推进的代数；
            - 也可直接构造 `GPConfig(population_size=..., max_generations=..., ...)` 观察不同预算。

            **运行入口**：接下来三个代码单元（`# E06`）。

            **源码位置**：`EvolutionEngine.initialize/step/run`；事件由 `apply_variation_traced` 与
            精英保留记录，经 `EvolutionTracer` 收集。

            > **我的预测（请填写）**：第 0 代最优个体的表达式是什么？前两代 best MSE 会明显下降吗？
            > “逐代 step”与“run_gp”的历史会有哪怕一位不同吗？精英保留事件里，子代表达式与父本相同吗？
        """),
        code(r"""
            # E06-A 初始化第 0 代并查看快照
            E06_SEED = 3
            e06_cfg = GPConfig(seed=E06_SEED)
            tracer = EvolutionTracer()
            engine = EvolutionEngine(e06_cfg, tracer)

            snap0 = engine.initialize()
            best0_idx = int(np.argmin(engine.fitnesses))
            print(f"第 0 代：population={len(engine.population)}，evaluations={engine.evaluations}")
            print(f"best 表达式：{engine.population[best0_idx].sexpr_str()}")
            print_table(
                ["快照字段", "值"],
                [
                    ("generation", snap0.generation),
                    ("best_train_mse", f"{snap0.best_train_mse:.6g}"),
                    ("best_test_mse", f"{snap0.best_test_mse:.6g}"),
                    ("mean / median / worst", f"{snap0.mean_fitness:.4g} / {snap0.median_fitness:.4g} / {snap0.worst_fitness:.4g}"),
                    ("mean_size / max_size", f"{snap0.mean_size:.2f} / {snap0.max_size}"),
                    ("mean_depth / max_depth", f"{snap0.mean_depth:.2f} / {snap0.max_depth}"),
                    ("unique_structures / ratio", f"{snap0.unique_structures} / {snap0.unique_ratio:.3f}"),
                    ("evaluations", snap0.evaluations),
                ],
            )
        """),
        code(r"""
            # E06-B 逐代推进：每代查看快照变化与真实事件
            E06_STEPS_BEFORE_PRINT = 2
            snapshot_rows = [(snap0.generation, f"{snap0.best_train_mse:.4g}",
                              f"{snap0.mean_fitness:.4g}", f"{snap0.median_fitness:.4g}",
                              f"{snap0.worst_fitness:.4g}", f"{snap0.mean_size:.1f}",
                              snap0.max_size, f"{snap0.unique_ratio:.3f}")]

            for _ in range(E06_STEPS_BEFORE_PRINT):
                snap = engine.step()
                snapshot_rows.append((snap.generation, f"{snap.best_train_mse:.4g}",
                                      f"{snap.mean_fitness:.4g}", f"{snap.median_fitness:.4g}",
                                      f"{snap.worst_fitness:.4g}", f"{snap.mean_size:.1f}",
                                      snap.max_size, f"{snap.unique_ratio:.3f}"))
                events = tracer.events_of_generation(snap.generation)
                counts = {}
                for event in events:
                    counts[event.event_type] = counts.get(event.event_type, 0) + 1
                print(f"\n第 {snap.generation} 代：完成 {len(events)} 个遗传事件 {counts}")
                for event_type in ("elitism", "crossover", "mutation", "reproduction"):
                    examples = [e for e in events if e.event_type == event_type]
                    if not examples:
                        continue
                    example = examples[0]
                    point = example.crossover_points if example.crossover_points is not None else example.mutation_point
                    print(f"  {event_type:<12} 父本下标={example.parent_indices} "
                          f"父本适应度={tuple(f'{v:.4g}' for v in (example.parent_fitnesses or ()))}")
                    print(f"  {'':<12} 点={point} fallback={example.fallback} changed={example.changed} "
                          f"子代适应度={tuple(f'{v:.4g}' for v in (example.child_fitnesses or ()))}")
                    print(f"  {'':<12} 父本表达式={example.parent_expressions[0][:80]}")
                    print(f"  {'':<12} 子代表达式={example.child_expressions[0][:80]}")

            print()
            print_table(
                ["gen", "best train", "mean", "median", "worst", "mean size", "max size", "unique ratio"],
                snapshot_rows,
            )
        """),
        code(r"""
            # E06-C 连续运行到停止 + 与一次性 run_gp 对比
            while not engine.done:
                engine.step()
            stepped = engine.result
            fresh = run_gp(e06_cfg)

            print(f"逐代运行停止原因：{stepped.stop_reason}；代数={stepped.n_generations}；"
                  f"评估次数={stepped.evaluations}")
            print(f"一次性运行停止原因：{fresh.stop_reason}；代数={fresh.n_generations}；"
                  f"评估次数={fresh.evaluations}")
            print()
            print_table(
                ["一致性检查", "结果"],
                [
                    ("best_mse_history 相同", stepped.best_mse_history == fresh.best_mse_history),
                    ("best_of_run 表达式相同", stepped.best_of_run.sexpr_str() == fresh.best_of_run.sexpr_str()),
                    ("evaluations 相同", stepped.evaluations == fresh.evaluations),
                    ("snapshots 代数相同", [s.generation for s in stepped.snapshots] == [s.generation for s in fresh.snapshots]),
                    ("Hall of Fame 长度", f"{len(stepped.hall_of_fame)}（两次运行一致：{len(stepped.hall_of_fame) == len(fresh.hall_of_fame)}）"),
                ],
            )
            print("\n最终表达式：", stepped.best_of_run.sexpr_str())
            print(f"训练 MSE={stepped.train_mse:.3e}，测试 MSE={stepped.test_mse:.3e}")
            print("\nHall of Fame（历代严格变优时追加的个体副本）：")
            for rank, tree in enumerate(stepped.hall_of_fame):
                print(f"  #{rank}: size={tree.size():3d} depth={tree.depth():2d} "
                      f"expr={tree.sexpr_str()[:70]}")
        """),
        code(r"""
            # E06-D 收敛曲线、适应度分布与多样性/复杂度演化
            snaps = stepped.snapshots
            gens = [s.generation for s in snaps]
            finite_or_nan = lambda values: [v if np.isfinite(v) else np.nan for v in values]

            fig, axes = plt.subplots(2, 2, figsize=(12, 6.4))

            axes[0, 0].semilogy(gens, finite_or_nan([s.best_train_mse for s in snaps]), "-o", ms=3, label="best train MSE")
            axes[0, 0].semilogy(gens, finite_or_nan([s.best_test_mse for s in snaps]), "--s", ms=3, label="best test MSE")
            axes[0, 0].axhline(e06_cfg.mse_target, color="tab:red", ls=":", lw=1, label="mse_target")
            axes[0, 0].set_xlabel("generation")
            axes[0, 0].set_ylabel("MSE (log)")
            axes[0, 0].set_title("best-of-generation: train vs test (test never feeds back)")
            axes[0, 0].legend(fontsize=7)

            axes[0, 1].semilogy(gens, finite_or_nan([s.mean_fitness for s in snaps]), label="mean fitness")
            axes[0, 1].semilogy(gens, finite_or_nan([s.median_fitness for s in snaps]), label="median fitness")
            axes[0, 1].semilogy(gens, finite_or_nan([s.worst_fitness for s in snaps]), label="worst fitness")
            axes[0, 1].set_xlabel("generation")
            axes[0, 1].set_ylabel("fitness (log)")
            axes[0, 1].set_title("Population fitness distribution")
            axes[0, 1].legend(fontsize=7)

            axes[1, 0].plot(gens, [s.unique_ratio for s in snaps], "-o", ms=3)
            axes[1, 0].set_xlabel("generation")
            axes[1, 0].set_ylabel("unique structures / population")
            axes[1, 0].set_title("Structural diversity")
            axes[1, 0].set_ylim(0, 1.05)

            axes[1, 1].plot(gens, [s.mean_size for s in snaps], "-o", ms=3, label="mean size")
            axes[1, 1].plot(gens, [s.max_size for s in snaps], "--s", ms=3, label="max size")
            axes[1, 1].set_xlabel("generation")
            axes[1, 1].set_ylabel("nodes")
            axes[1, 1].set_title("Tree size (bloat check)")
            axes[1, 1].legend(fontsize=7)

            fig.tight_layout()
            print("注意：纵轴使用对数；inf 值被画成断点（数值异常个体被策略固定为 inf）。")
        """),
        md(r"""
            ### 观察与解释（请填写）

            > **观察与解释（请填写）**：第 0 代到停止，best MSE 是否单调不增？mean/median 是否也单调？
            > 结构多样性如何变化？平均树规模有没有上升？逐代与批量的五项一致性检查是否全部通过？
            > 从真实事件看，第 1 代精英保留的子代表达式与父本是否完全相同？

            **本章局限**：只运行了一个种子；收敛曲线平滑不代表算法稳健。`mean_fitness` 可能被
            极少数超大树拉到极大值甚至 inf，因此对数坐标下会出现断点——这本身是“重尾/膨胀”的线索，
            但需要 Chapter 8 的对照实验才能判断机制。

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


# ---------------------------------------------------------------------------
# Chapter 7
# ---------------------------------------------------------------------------
def chapter_7() -> list:
    return [
        md(r"""
            ## Chapter 7：随机性与搜索预算

            > 第二篇导言：第一篇把 GP 的每个阶段单独拆开看了一遍；从本章开始处理“为什么有时失败”。
            > 每一章都要求先写预测，再用**预先指定种子**的对照实验检验，而不是用一个成功/失败案例
            > 讲故事。

            **核心问题**：GP 的随机性来自哪里？预算怎样限制搜索？不同随机种子的结果分布说明了什么？

            ### 概念与数学定义

            **随机性来源**（本项目全部来自同一个 `random.Random(config.seed)`）：

            1. 初始化：深度桶、full/grow 选择、节点与运算符抽样；
            2. 选择：每轮锦标赛抽到的 $k$ 个下标；
            3. 交叉：两个交叉点；
            4. 变异：变异点与 grow 新子树；
            5. 复制/精英：不消耗随机数（但会改变种群构成）。

            **预算**：用**训练适应度评估次数**衡量：

            $$\text{evaluations}=N_{\text{pop}}\times(\text{评价过的代数}).$$

            它比墙钟时间稳定，也不受机器性能影响。测试集评价不计入预算，也不反馈给搜索。

            **首达代**：$\min\{g:\ \mathrm{MSE}_{\text{train}}(g)<\text{mse\_target}\}$；若直到预算用完
            都没达到则为 `None`。**成功率**必须说明是训练达标率还是测试达标率。

            **重复性**：同一 seed 与配置逐位可复现；不同 seed 是同一随机过程的不同样本。只报告一个
            成功种子会系统性高估算法能力——这正是旧版种子搜索的做法，必须明确它的教学性质与偏差。

            ### 具体小例子

            若 6 个种子里只有 2 个训练达标，则训练达标率是 $2/6$；它们的首达代可能分别是 3 和 7。
            如果其中一个种子的测试 MSE 很大，测试达标率会更低。**这两个比例不能混为一谈。**

            ### 源码入口

            - 引擎的预算与停止：`gptoy/evolution.py` → `EvolutionEngine.evaluations`、`stop_reason`
            - 批量与汇总：`gptoy/experiments.py` → `run_seed_batch`、`run_budget_experiment`、`summarize_runs`
            - 随机源：`gptoy/utils.py` → `make_rng`
        """),
        md(r"""
            ### E07：多种子 × 多预算

            **研究问题**：在预先指定的种子集合上，不同搜索预算下的最终训练/测试 MSE、训练/测试达标率、
            首达代与实际评估次数如何分布？种子之间的差异有多大？

            **核心概念**：种子分布、预算、首达代、训练/测试达标率、结构恢复代理。

            **可修改配置**（改下面单元顶部变量后重跑）：

            - `E07_SEEDS`：**预先指定**的种子（不要看结果再换）；
            - `E07_BUDGETS`：最大代数预算；
            - `E07_POP_SIZE`、`E07_MAX_DEPTH`：种群与深度上限（调小是为了在几十秒内跑完）。

            **运行入口**：接下来两个代码单元（`# E07`）。

            **源码位置**：`run_budget_experiment` → `run_seed_batch` → 同一 `EvolutionEngine`；
            汇总用 `summarize_runs`（结构恢复率默认 `None`，因为这里没有可靠的结构判别器）。

            > **我的预测（请填写）**：预算从 5 代增加到 15 代，训练达标率会提高吗？测试达标率呢？
            > 种子之间的最终 MSE 会差几个数量级？首达代的分布是集中还是分散？实际评估次数等于
            > 种群规模 × (预算+1) 吗？
        """),
        code(r"""
            # E07 预先指定种子与预算，保留全部结果
            E07_SEEDS = (0, 1, 2, 3, 4, 5)
            E07_BUDGETS = (5, 15)
            E07_POP_SIZE = 40
            E07_MAX_DEPTH = 9

            e07_base = replace(cfg, population_size=E07_POP_SIZE, max_tree_depth=E07_MAX_DEPTH,
                               stop_on_target=False)  # 关闭提前停止，预算才会被真正消耗
            e07_results = run_budget_experiment(e07_base, E07_SEEDS, E07_BUDGETS)

            rows = []
            for budget, results in e07_results.items():
                for r in results:
                    rows.append((budget, r.seed, f"{r.train_mse:.3e}", f"{r.test_mse:.3e}",
                                 r.generation_found, r.n_generations, r.evaluations))
            print_table(["budget", "seed", "final train MSE", "final test MSE", "first hit gen",
                         "generations", "evaluations"], rows)
            print("\n说明：测试 MSE 只用于最终报告，没有参与任何选择或种子筛选。")
        """),
        code(r"""
            # E07 汇总：训练/测试达标率、首达代、评估次数（全部种子，不筛选）
            print_table(
                ["budget", "train 达标率", "test 达标率", "首达代 values", "首达代中位数",
                 "平均评估次数", "失败种子（训练）"],
                [
                    (budget,
                     f"{summary['train_success_rate']:.2f}",
                     f"{summary['test_success_rate']:.2f}",
                     summary['first_hit_generation']['values'],
                     summary['first_hit_generation']['median'],
                     f"{summary['evaluations']['mean']:.0f}",
                     summary['failed_train_seeds'])
                    for budget, results in e07_results.items()
                    for summary in [summarize_runs(results)]
                ],
            )
            print()
            for budget, results in e07_results.items():
                summary = summarize_runs(results)
                print(f"budget={budget}: 结构恢复率 = {summary['structural_recovery_rate']} "
                      f"（{summary['structural_recovery_note']}）")
        """),
        code(r"""
            # E07 可视化：种子差异（每个点是一个真实种子）
            fig, axes = plt.subplots(1, 2, figsize=(12, 3.6))
            colors = {5: "tab:orange", 15: "tab:blue"}

            for budget, results in e07_results.items():
                seeds = [r.seed for r in results]
                train = [max(r.train_mse, 1e-32) for r in results]
                test = [max(r.test_mse, 1e-32) for r in results]
                axes[0].scatter(seeds, train, label=f"budget={budget}", color=colors[budget], s=45)
                axes[1].scatter(seeds, test, label=f"budget={budget}", color=colors[budget], s=45)

            axes[0].axhline(e07_base.mse_target, color="tab:red", ls="--", lw=1, label="mse_target")
            axes[0].set_yscale("log")
            axes[0].set_xticks(E07_SEEDS)
            axes[0].set_xlabel("seed")
            axes[0].set_ylabel("final train MSE (log)")
            axes[0].set_title("Final train MSE per seed (all seeds shown)")
            axes[0].legend(fontsize=7)

            axes[1].axhline(e07_base.mse_target, color="tab:red", ls="--", lw=1, label="mse_target")
            axes[1].set_yscale("log")
            axes[1].set_xticks(E07_SEEDS)
            axes[1].set_xlabel("seed")
            axes[1].set_ylabel("final test MSE (log)")
            axes[1].set_title("Final test MSE per seed (never used for selection)")
            axes[1].legend(fontsize=7)
            fig.tight_layout()

            print("每个点是一个预先指定的种子；如果只展示最好的种子，就会隐藏这里看到的大部分失败。")
        """),
        md(r"""
            ### 观察与解释（请填写）

            > **观察与解释（请填写）**：训练达标率与测试达标率分别是多少？两者差距说明什么？
            > 首达代分布集中在哪个范围？预算翻倍后，失败种子变少了吗？实际评估次数与
            > `种群规模 × (预算+1)` 是否一致？种子之间的最终 MSE 差几个数量级？
            > 如果只能用训练集信息做模型选择，你会怎么选？测试集为什么不能参与？

            **本章诚实性说明**：

            - 种子集合是**事先指定**的，没有根据测试表现增删；
            - 这里的预算（5/15 代、种群 40）远小于默认配置，失败率被刻意放大，便于观察分布；
            - “结构恢复率”没有计算，因为本实验没有可靠的全域结构判别器；把它写成 0 或 100% 都是误导；
            - 默认基线使用的 `seed=3` 来自旧版 `scripts/find_seed.py` 的教学种子搜索。该脚本先用
              训练+测试同时达标的规则筛选种子，再按首达代排序，因此它展示的是“被选中的成功样本”，
              **不能**用来估计无偏成功率；真正的成功率请看本实验的多种子统计。

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
