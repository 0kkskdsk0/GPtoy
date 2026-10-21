"""GPtoy 学习实验室内容（第三部分）：Chapter 8–11 与附录。"""

from lab_content import code, md

CHAPTERS_C = [
    {
        "num": 8,
        "title": "早熟收敛与表达式膨胀",
        "part": 1,
        "core_question": "选择压力变大会不会让种群过早失去多样性？表达式复杂度会不会无界增长？怎样设计对照实验来检验？",
        "source": "`gptoy/config.py`（`stop_on_target`）、`gptoy/selection.py`、`gptoy/metrics.py`（`population_stats`）、`gptoy/experiments.py`（`make_parsimony_fitness`）、`gptoy/evolution.py`（`EvolutionEngine` 的 `fitness_fn` 参数）",
        "experiments": ["E08"],
        "concepts": [
            "早熟收敛：种群多样性快速下降、best 长期停滞，但尚未达到预算/全局最优",
            "多样性指标：结构唯一比例 unique_ratio、适应度标准差、深度/规模分布",
            "膨胀（bloat）：平均/最大树规模持续增长，但适应度没有相应改善",
            "简约压力（parsimony pressure）：用 $\\mathrm{MSE}+\\lambda\\cdot\\mathrm{size}$ 作为搜索目标，惩罚大 $\\lambda$ 时选择偏好小树",
            "复杂度限制：直接降低 max_tree_depth 会改变可达结构集合，是另一种控制手段",
            "不预设结论：k 大不一定早熟，运行也不一定膨胀；必须看对照实验的区分能力数据",
        ],
    },
    {
        "num": 9,
        "title": "小样本、噪声与泛化",
        "part": 1,
        "core_question": "训练样本变少或噪声变大时，GP 的泛化会怎样变化？训练/验证/测试应该怎样划分？",
        "source": "`gptoy/experiments.py`（`make_train_val_test`、`add_gaussian_noise`、`summarize_runs`）、`gptoy/evolution.py`（`data` 参数）、`gptoy/operators.py`（`fitness`）",
        "experiments": ["E09"],
        "concepts": [
            "过拟合：训练误差低但独立测试误差高；在 GP 中常表现为利用样本噪声的复杂结构",
            "泛化：在独立同分布（或明确外推区间）数据上的表现",
            "样本量：训练点越少，同一函数类中与训练数据一致的候选越多（欠定）",
            "观测噪声：$y_i=f(x_i)+\\varepsilon_i$，$\\varepsilon_i\\sim N(0,\\sigma^2)$，$\\sigma=\\text{noise\\_ratio}\\cdot\\mathrm{std}(y)$（ddof=0）",
            "数据划分：训练/验证/测试的 x 区间互不重叠；测试只做最终评价；测试区间在训练区间之外，故度量的是外推（extrapolation）误差，不是 i.i.d. 泛化误差",
            "模型选择偏差：用测试集挑种子/调参会把测试误差变成乐观偏置",
        ],
    },
    {
        "num": 10,
        "title": "搜索空间与数值语义的限制",
        "part": 1,
        "core_question": "目标是否“可表达”与有限预算下能否搜到是同一件事吗？保护除法与数值异常怎样改变语义？",
        "source": "`gptoy/data.py`（`make_target_data`）、`gptoy/operators.py`（`p_div`、`evaluate`、`fitness`）、`gptoy/evolution.py`（`data` 参数）、`gptoy/metrics.py`",
        "experiments": ["E10"],
        "concepts": [
            "可达性：目标函数是否在由函数集/终止符集生成的表达式空间中",
            "算子集限制：sin 不是有理函数，仅用 $\\{+,-,*,/\\}$ 与 x 的有限表达式无法精确表示 sin",
            "表达力 vs 可搜索性：目标可表达不代表有限预算内能搜到；不可精确表达也不代表不能在有限样本上近似",
            "常数表示：本项目没有随机常数，$1$、$0$ 只能由 $x/x$、$x-x$ 合成，系数搜索更困难",
            "保护除法：分母为 0 时返回 1.0，改变语义；近零分母会放大数值误差；深树可能溢出为 inf",
            "数值异常策略：`fitness` 对非有限预测返回 inf，不产生 NaN 污染选择；默认 seed=3 轨迹不受影响",
        ],
    },
    {
        "num": 11,
        "title": "GP 失败机制与改进方法",
        "part": 2,
        "core_question": "如何把观察到的失败现象组织成可检验的机制解释？前人可能的改进分别针对哪些机制？",
        "source": "综合 `gptoy/` 全部模块与 Chapter 0–10 的实验结果；重点 `gptoy/evolution.py`、`gptoy/metrics.py`、`gptoy/tracing.py`、`gptoy/experiments.py`",
        "experiments": ["E01", "E02", "E03", "E04", "E05", "E06", "E07", "E08", "E09", "E10"],
        "concepts": [
            "失败分类至少 7 类：搜索效率不足、多样性丧失/停滞、复杂度膨胀、噪声/过拟合、表达式空间限制、常数优化困难、评价指标/模型选择问题",
            "每类失败需要：机制假设、可观察指标、支持/反对证据、替代解释、可调研的改进方向",
            "证据分级：已观察现象 / 对照实验支持的机制 / 未检验的猜测 / 外部文献与数学事实",
            "研究问题来自“现象 + 可区分预测”，而不是“我知道某个算法更好”",
            "改进方向必须对应机制：增加多样性、控制复杂度、改变表示、改进评价、混合局部搜索等",
        ],
    },
]


# ---------------------------------------------------------------------------
# Chapter 8
# ---------------------------------------------------------------------------
def chapter_8() -> list:
    return [
        md(r"""
            ## Chapter 8：早熟收敛与表达式膨胀

            **核心问题**：选择压力变大会不会让种群过早失去多样性？表达式复杂度会不会无界增长？怎样设计对照实验来检验？

            ### 概念与数学定义

            **早熟收敛（premature convergence）**：种群在尚未耗尽预算、也未必达到全局最优时，
            结构多样性快速下降、best 适应度长期停滞。注意它描述的是**过程**，不是“结果不够好”：
            一个已经收敛到正确公式的运行也会失去多样性，那是成功收敛。

            **多样性指标**（本项目可观察）：

            - `unique_ratio = 唯一结构数 / 种群规模`（基于结构签名，见 `metrics.structural_signature`）；
            - `fitness_std`：当代适应度的总体标准差；
            - `mean_depth` / `max_depth`、`mean_size` / `max_size` 的逐代变化。

            **膨胀（bloat）**：平均/最大树规模持续增长，而 best 适应度没有相应改善。常见机制假设是
            “稍大的树偶尔能拟合得更好，选择因此间接奖励规模增长”，但本实验不预设它一定发生。

            **简约压力（parsimony pressure）**：把搜索目标改成

            $$\text{objective}(T)=\mathrm{MSE}(T)+\lambda\cdot\mathrm{size}(T).$$

            $\lambda=0$ 时与原目标完全一致；$\lambda>0$ 时选择同时偏好小树。注意这是**实验变体**，
            不是原 Koza-1 算法；快照中的 `best_train_mse` 仍是原始 MSE，因此可以观察“误差-复杂度”权衡。

            **复杂度限制**：直接降低 `max_tree_depth` 会改变可达结构集合，可能抑制膨胀，也可能
            同时砍掉有用的深结构——这是需要实验区分的替代解释。

            ### 具体小例子

            若某配置的 `unique_ratio` 从 0.95 在 5 代内降到 0.2，同时 best MSE 不再下降，则它表现出
            “多样性丧失 + 停滞”的**现象**；但停滞也可能来自“搜索空间里没有更好的可达结构”“预算不足”
            或“评价噪声”，需要额外对照才能区分机制。

            ### 源码入口

            - 关闭提前停止：`GPConfig(stop_on_target=False)`；`run_gp(..., max_generations=...)`
            - 选择压力：`gptoy/selection.py` → `tournament_select`
            - 每代统计：`gptoy/metrics.py` → `population_stats`
            - 简约目标：`gptoy/experiments.py` → `make_parsimony_fitness`
            - 自定义搜索目标：`EvolutionEngine(config, fitness_fn=...)` / `run_gp(..., fitness_fn=...)`
        """),
        md(r"""
            ### E08：两组对照实验

            **研究问题**：

            - A：只改变锦标赛规模 $k\in\{1,7,12\}$，选择压力与搜索过程（best MSE、多样性、规模）如何变化？
            - B：改变复杂度限制（`max_tree_depth`）或加入简约惩罚 $\lambda$，复杂度的发展如何变化？

            **核心概念**：选择压力、多样性、膨胀、简约压力、复杂度限制、关闭提前停止。

            **可修改配置**（改下面单元顶部变量后重跑）：

            - `E08A_KS`、`E08A_SEEDS`、`E08A_POP_SIZE`、`E08A_GENERATIONS`；
            - `E08B_LAMBDAS`、`E08B_DEPTHS`、`E08B_SEED`。

            **运行入口**：接下来三个代码单元（`# E08-A`、`# E08-B`）。

            **源码位置**：`run_gp` 的 `stop_on_target=False` 用于跑满预算；
            `make_parsimony_fitness(λ)` 提供简约目标。

            > **我的预测（请填写）**：k 增大时，`unique_ratio` 会更快下降吗？平均树规模会更大还是更小？
            > 加入 $\lambda$ 后，最终原始 MSE 会变差吗？规模会变小吗？降低 `max_tree_depth` 与
            > 加入简约惩罚，哪一个更能抑制规模？先写下判断，再用下面的数据检验。
        """),
        code(r"""
            # E08-A 只改变 tournament size：关闭提前停止，跑满预算
            E08A_KS = (1, 7, 12)
            E08A_SEEDS = (0, 1)
            E08A_POP_SIZE = 40
            E08A_GENERATIONS = 12
            E08A_DEPTH = 9

            e08a_base = replace(cfg, population_size=E08A_POP_SIZE,
                                max_generations=E08A_GENERATIONS,
                                max_tree_depth=E08A_DEPTH, stop_on_target=False)
            e08a = {
                k: [run_gp(replace(e08a_base, tournament_size=k, seed=seed)) for seed in E08A_SEEDS]
                for k in E08A_KS
            }

            print_table(
                ["k", "seed", "final train MSE", "test MSE", "unique ratio (last)",
                 "mean size (last)", "max size (last)", "evaluations"],
                [
                    (k, r.seed, f"{r.train_mse:.4g}", f"{r.test_mse:.4g}",
                     f"{r.snapshots[-1].unique_ratio:.3f}", f"{r.snapshots[-1].mean_size:.1f}",
                     r.snapshots[-1].max_size, r.evaluations)
                    for k, runs in e08a.items() for r in runs
                ],
            )
        """),
        code(r"""
            # E08-A 逐代曲线：每个 k 的两种子细线 + 均值粗线
            fig, axes = plt.subplots(1, 3, figsize=(15, 3.6))
            for k, runs in e08a.items():
                for r in runs:
                    axes[0].semilogy([max(v, 1e-32) for v in r.best_mse_history], color=f"C{k % 10}", alpha=0.35, lw=1)
                    axes[1].plot([s.unique_ratio for s in r.snapshots], color=f"C{k % 10}", alpha=0.35, lw=1)
                    axes[2].plot([s.mean_size for s in r.snapshots], color=f"C{k % 10}", alpha=0.35, lw=1)
                best_curve = np.mean([[max(v, 1e-32) for v in r.best_mse_history] for r in runs], axis=0)
                unique_curve = np.mean([[s.unique_ratio for s in r.snapshots] for r in runs], axis=0)
                size_curve = np.mean([[s.mean_size for s in r.snapshots] for r in runs], axis=0)
                axes[0].semilogy(best_curve, color=f"C{k % 10}", lw=2.4, label=f"k={k}")
                axes[1].plot(unique_curve, color=f"C{k % 10}", lw=2.4, label=f"k={k}")
                axes[2].plot(size_curve, color=f"C{k % 10}", lw=2.4, label=f"k={k}")

            axes[0].axhline(e08a_base.mse_target, color="tab:red", ls="--", lw=1)
            axes[0].set_xlabel("generation")
            axes[0].set_ylabel("best train MSE (log)")
            axes[0].set_title("Best MSE (thin: each seed, thick: mean)")
            axes[0].legend(fontsize=7)
            axes[1].set_xlabel("generation")
            axes[1].set_ylabel("unique ratio")
            axes[1].set_title("Structural diversity")
            axes[1].set_ylim(0, 1.05)
            axes[1].legend(fontsize=7)
            axes[2].set_xlabel("generation")
            axes[2].set_ylabel("mean tree size (nodes)")
            axes[2].set_title("Mean tree size over generations")
            axes[2].legend(fontsize=7)
            fig.tight_layout()
        """),
        code(r"""
            # E08-A 每代明细：best/mean/median/worst 适应度、结构多样性、平均/最大规模
            # （选一个代表性配置 k=7、seed=0；其他配置可用同样方式查看）
            detail = e08a[7][0]
            print_table(
                ["gen", "best train", "mean", "median", "worst", "unique ratio", "mean size", "max size"],
                [
                    (s.generation, f"{s.best_train_mse:.4g}", f"{s.mean_fitness:.4g}",
                     f"{s.median_fitness:.4g}", f"{s.worst_fitness:.4g}",
                     f"{s.unique_ratio:.3f}", f"{s.mean_size:.1f}", s.max_size)
                    for s in detail.snapshots
                ],
            )
            print("\n（这张表就是“每代记录 best MSE、适应度分布、多样性、平均/最大规模”的入口；")
            print("  `GenerationSnapshot` 的字段可直接用于你自己的表格或图。）")
        """),
        code(r"""
            # E08-B 复杂度限制 vs 简约惩罚
            E08B_SEED = 3
            E08B_POP_SIZE = 40
            E08B_GENERATIONS = 15
            E08B_DEPTHS = {"depth<=5": 5, "depth<=9": 9}
            E08B_LAMBDAS = {"MSE+0.001*size": 0.001, "MSE+0.01*size": 0.01}

            e08b_base = replace(cfg, population_size=E08B_POP_SIZE, max_generations=E08B_GENERATIONS,
                                max_tree_depth=9, stop_on_target=False, seed=E08B_SEED)
            e08b = {}
            for name, depth in E08B_DEPTHS.items():
                e08b[name] = run_gp(replace(e08b_base, max_tree_depth=depth))
            for name, lam in E08B_LAMBDAS.items():
                fitness_fn = make_parsimony_fitness(lam) if lam > 0 else None
                e08b[name] = run_gp(e08b_base, fitness_fn=fitness_fn)

            print_table(
                ["variant", "raw train MSE", "test MSE", "final size", "final depth",
                 "mean size (last)", "max size (last)", "unique ratio (last)"],
                [
                    (name, f"{r.train_mse:.4g}", f"{r.test_mse:.4g}", r.best_of_run.size(),
                     r.best_of_run.depth(), f"{r.snapshots[-1].mean_size:.1f}",
                     r.snapshots[-1].max_size, f"{r.snapshots[-1].unique_ratio:.3f}")
                    for name, r in e08b.items()
                ],
            )
            print()
            print("区分能力检查（本实验内部极差）：")
            print(f"  unique_ratio 极差 = {max(r.snapshots[-1].unique_ratio for r in e08b.values()) - min(r.snapshots[-1].unique_ratio for r in e08b.values()):.3f}")
            print(f"  mean_size 极差    = {max(r.snapshots[-1].mean_size for r in e08b.values()) - min(r.snapshots[-1].mean_size for r in e08b.values()):.1f}")
            print(f"  max_size 极差     = {max(r.snapshots[-1].max_size for r in e08b.values()) - min(r.snapshots[-1].max_size for r in e08b.values())}")
            print(f"  raw train MSE 极差= {max(r.train_mse for r in e08b.values()) - min(r.train_mse for r in e08b.values()):.4g}")
        """),
        code(r"""
            # E08-B 可视化：规模发展与最终误差
            fig, axes = plt.subplots(1, 3, figsize=(15, 3.6))
            for name, r in e08b.items():
                gens = [s.generation for s in r.snapshots]
                axes[0].plot(gens, [s.mean_size for s in r.snapshots], "-o", ms=3, label=name)
                axes[1].plot(gens, [s.max_size for s in r.snapshots], "-s", ms=3, label=name)
            axes[0].set_xlabel("generation")
            axes[0].set_ylabel("mean size (nodes)")
            axes[0].set_title("Mean tree size")
            axes[0].legend(fontsize=7)
            axes[1].set_xlabel("generation")
            axes[1].set_ylabel("max size (nodes)")
            axes[1].set_title("Max tree size")
            axes[1].legend(fontsize=7)

            positions = np.arange(len(e08b))
            axes[2].bar(positions, [max(r.train_mse, 1e-33) for r in e08b.values()], color="tab:purple")
            axes[2].set_yscale("log")
            axes[2].set_xticks(positions)
            axes[2].set_xticklabels(list(e08b), rotation=25, ha="right", fontsize=7)
            axes[2].set_ylabel("final raw train MSE (log)")
            axes[2].set_title("Final raw train MSE (not penalized)")
            fig.tight_layout()
        """),
        md(r"""
            ### 观察与解释（请填写）

            > **观察与解释（请填写）**：A 组中 k 增大时，`unique_ratio` 的下降速度、平均规模和最终
            > best MSE 有什么差别？这些差别在两个种子上一致吗？B 组中哪种手段让规模更小？代价是
            > 原始 MSE 变差吗？是否存在“规模变大但误差没有改善”的膨胀现象？
            > 如果**没有**观察到预期现象，请如实写下，并引用“区分能力检查”的数字说明实验是否仍有
            > 分辨力（例如极差是否接近 0）。

            **本章诚实性要求**：本 Notebook **不预设** k 越大越早熟、也不预设一定发生膨胀。
            上面的数字来自本次运行；两个种子只是很小的样本，不能据此做普遍性结论。请把
            “本次观察到……”“我推测的机制是……”“还需要什么对照才能区分……”分开写。

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
# Chapter 9
# ---------------------------------------------------------------------------
def chapter_9() -> list:
    return [
        md(r"""
            ## Chapter 9：小样本、噪声与泛化

            **核心问题**：训练样本变少或噪声变大时，GP 的泛化会怎样变化？训练/验证/测试应该怎样划分？

            ### 概念与数学定义

            **生成机制**：本实验的数据来自已知目标函数 $f$，训练标签加观测噪声：

            $$y_i = f(x_i)+\varepsilon_i,\qquad \varepsilon_i \overset{iid}{\sim} N(0,\sigma^2),$$

            $$\sigma = \text{noise\_ratio}\times \mathrm{std}\big(f(x_{\text{train}})\big),$$

            其中 `std` 使用**总体标准差（ddof=0）**。噪声随机源由 `data_seed` 单独控制，与 GP 的
            `seed` 分离，因此可以固定数据、只改变搜索随机性，反之亦然。

            **数据划分**：三个 x 区间互不重叠（默认训练 $[-1,0.9]$、验证 $[1.0,1.4]$、测试 $[1.5,2.0]$），
            验证/测试标签保持干净（不加噪声）。测试集只在最后评价。

            **注意这是外推（extrapolation）**：测试区间 $[1.5,2.0]$ 完全位于训练区间 $[-1,0.9]$ 之外，
            验证区间 $[1.0,1.4]$ 也在训练区间之外。因此本实验的“干净测试 MSE”主要度量**外推误差**，
            而不是同分布 i.i.d. 泛化误差；若要估计后者，需要在训练分布内部做独立划分。

            **过拟合**：训练误差低但独立测试误差高。GP 中的常见形式是选择出利用训练噪声的复杂结构；
            但在小预算下也可能根本拟合不了，表现为训练/测试都差——两种失败要分开看。

            **模型选择偏差**：如果用测试集挑种子、挑超参数或挑模型，那么报告出来的测试误差会系统性
            偏乐观。正确做法是：用训练/验证做选择，测试只做一次最终评价。

            ### 具体小例子

            若训练集只有 10 个点、噪声 $\sigma\approx0.1$，则即使找到真实函数，训练 MSE 也会在
            $\sigma^2\approx0.01$ 附近；此时“训练 MSE 达标 1e-6”本身就不可能。**阈值必须和噪声尺度
            一起解释。**

            ### 源码入口

            - 数据划分与噪声：`gptoy/experiments.py` → `make_train_val_test`、`add_gaussian_noise`
            - 用自定义数据训练：`run_gp(config, data=split)` / `EvolutionEngine(config, data=...)`
            - 复算验证/测试误差：`gptoy/operators.py` → `fitness`
        """),
        md(r"""
            ### E09：样本量 × 噪声的 2×2 对照

            **研究问题**：在训练样本量（10/20）与噪声水平（0/0.1）的 2×2 组合下，训练/验证/干净测试
            （外推）MSE、训练-外推差距与最终复杂度如何变化？

            **核心概念**：过拟合、泛化、样本量、观测噪声、数据划分、模型选择偏差。

            **可修改配置**（改下面单元顶部变量后重跑）：

            - `E09_N_TRAIN`、`E09_NOISE_RATIOS`：2×2 网格；
            - `E09_GP_SEEDS`：每个条件下重复的 GP 种子（不要按测试结果挑选）；
            - `E09_DATA_SEED`：控制噪声的数据随机种子（与 GP 种子分离）。

            **运行入口**：接下来三个代码单元（`# E09`）。

            **源码位置**：`make_train_val_test`、`add_gaussian_noise`；训练循环仍是同一个引擎。

            > **我的预测（请填写）**：噪声为 0.1 时训练 MSE 的下限大约是多少？样本从 20 减到 10
            > 会让测试 MSE 变差吗？在 10 个点 + 噪声时，GP 更可能找到真实结构还是记住噪声？
            > 训练-测试差距在哪种条件下最大？
        """),
        code(r"""
            # E09 2x2 对照：样本量 x 噪声；同一划分下重复两个 GP 种子
            E09_N_TRAIN = (10, 20)
            E09_NOISE_RATIOS = (0.0, 0.1)
            E09_GP_SEEDS = (1, 2)
            E09_DATA_SEED = 7
            E09_POP_SIZE = 40
            E09_GENERATIONS = 20
            E09_MAX_DEPTH = 9

            e09_rows = []
            for n_train in E09_N_TRAIN:
                for noise_ratio in E09_NOISE_RATIOS:
                    split = make_train_val_test(
                        n_train=n_train, n_val=10, n_test=10,
                        noise_ratio=noise_ratio, seed=E09_DATA_SEED,
                    )
                    for gp_seed in E09_GP_SEEDS:
                        run_cfg = replace(cfg, population_size=E09_POP_SIZE,
                                          max_generations=E09_GENERATIONS,
                                          max_tree_depth=E09_MAX_DEPTH,
                                          stop_on_target=False, seed=gp_seed)
                        result = run_gp(run_cfg, data=split)
                        val_mse = fitness(result.best_of_run, split["x_val"], split["y_val"])
                        e09_rows.append({
                            "n_train": n_train,
                            "noise": noise_ratio,
                            "sigma": split["sigma"],
                            "gp_seed": gp_seed,
                            "train": result.train_mse,
                            "val": val_mse,
                            "test": result.test_mse,
                            "size": result.best_of_run.size(),
                        })

            print_table(
                ["n_train", "noise", "sigma", "gp_seed", "train MSE", "val MSE",
                 "clean test MSE (extrapolation)", "test-train gap (extrapolation)", "final size"],
                [
                    (r["n_train"], r["noise"], f"{r['sigma']:.4g}", r["gp_seed"],
                     f"{r['train']:.3e}", f"{r['val']:.3e}", f"{r['test']:.3e}",
                     f"{r['test'] - r['train']:.3e}", r["size"])
                    for r in e09_rows
                ],
            )
        """),
        code(r"""
            # E09 条件汇总（每个条件两个 GP 种子都保留）
            conditions = []
            for n_train in E09_N_TRAIN:
                for noise_ratio in E09_NOISE_RATIOS:
                    subset = [r for r in e09_rows if r["n_train"] == n_train and r["noise"] == noise_ratio]
                    conditions.append((n_train, noise_ratio, subset))

            print_table(
                ["n_train", "noise", "train MSE (mean)", "val MSE (mean)",
                 "clean test MSE (extrapolation, mean)", "test-train gap (mean)", "size (mean)"],
                [
                    (n_train, noise,
                     f"{np.mean([r['train'] for r in subset]):.3e}",
                     f"{np.mean([r['val'] for r in subset]):.3e}",
                     f"{np.mean([r['test'] for r in subset]):.3e}",
                     f"{np.mean([r['test'] - r['train'] for r in subset]):.3e}",
                     f"{np.mean([r['size'] for r in subset]):.1f}")
                    for n_train, noise, subset in conditions
                ],
            )

            # 模型选择只使用验证 MSE；测试 MSE 只作为最终报告，不参与任何 min/排序
            best_by_val = min(conditions, key=lambda item: np.mean([r["val"] for r in item[2]]))
            chosen_test = np.mean([r["test"] for r in best_by_val[2]])
            print()
            print(f"仅用验证 MSE 选择的配置：n_train={best_by_val[0]}, noise={best_by_val[1]}；"
                  f"其干净测试 MSE（外推）均值={chosen_test:.3e}")
            print("测试 MSE 没有参与上面的比较或排序；它只用于最终报告。")
        """),
        code(r"""
            # E09 可视化：训练 vs 干净外推测试（log-log），以及各条件测试 MSE
            fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))
            markers = {10: "o", 20: "s"}
            colors = {0.0: "tab:blue", 0.1: "tab:orange"}
            for n_train in E09_N_TRAIN:
                for noise_ratio in E09_NOISE_RATIOS:
                    subset = [r for r in e09_rows if r["n_train"] == n_train and r["noise"] == noise_ratio]
                    axes[0].scatter([max(r["train"], 1e-33) for r in subset],
                                    [max(r["test"], 1e-33) for r in subset],
                                    marker=markers[n_train], color=colors[noise_ratio], s=55,
                                    label=f"n={n_train}, noise={noise_ratio}")
            lower = 1e-33
            axes[0].plot([lower, 1e3], [lower, 1e3], "k--", lw=1, label="extrapolated test = train")
            axes[0].set_xscale("log")
            axes[0].set_yscale("log")
            axes[0].set_xlabel("train MSE (log)")
            axes[0].set_ylabel("clean test MSE (extrapolation, log)")
            axes[0].set_title("Train vs clean-test (extrapolation) MSE (each point = one GP seed)")
            axes[0].legend(fontsize=6)

            positions = np.arange(len(conditions))
            means = [np.mean([r["test"] for r in subset]) for _, _, subset in conditions]
            axes[1].bar(positions, [max(v, 1e-33) for v in means], color="tab:green", alpha=0.8)
            for position, (_, _, subset) in zip(positions, conditions):
                axes[1].scatter([position] * len(subset), [max(r["test"], 1e-33) for r in subset],
                                color="black", s=18, zorder=3)
            axes[1].set_yscale("log")
            axes[1].set_xticks(positions)
            axes[1].set_xticklabels([f"n={n}\nnoise={noise}" for n, noise, _ in conditions], fontsize=7)
            axes[1].set_ylabel("clean test MSE (extrapolation, log)")
            axes[1].set_title("Bars = mean over GP seeds; dots = individual seeds")
            fig.tight_layout()
        """),
        md(r"""
            ### 观察与解释（请填写）

            > **观察与解释（请填写）**：噪声为 0.1 时训练 MSE 的下限是否接近 $\sigma^2$？
            > 样本量从 20 减到 10 对外推测试 MSE 有什么影响？有没有“训练 MSE 很低但外推测试 MSE
            > 很高”的过拟合迹象？训练-外推差距在哪个条件下最大？两个 GP 种子的结论一致吗？
            > 用验证集选出的是哪个配置，它的外推表现如何？如果改用测试集来选择，会引入什么偏差？

            **本章诚实性说明**：

            - 训练/验证/测试的 x 区间互不重叠；测试标签始终干净；
            - 测试区间 $[1.5,2.0]$ 在训练区间 $[-1,0.9]$ 之外，故报告的是**外推误差**，不是 i.i.d. 泛化误差；
            - 测试集没有参与搜索、调参或种子挑选；
            - 噪声由 `data_seed=7` 固定，GP 搜索种子另行指定（1, 2），两者分离；
            - 每个条件只有 2 个 GP 种子，结论只适用于本次观察，不外推为普遍规律。

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
# Chapter 10
# ---------------------------------------------------------------------------
def chapter_10() -> list:
    return [
        md(r"""
            ## Chapter 10：搜索空间与数值语义的限制

            **核心问题**：目标是否“可表达”与有限预算下能否搜到是同一件事吗？保护除法与数值异常怎样改变语义？

            ### 概念与数学定义

            **可达性**：给定函数集 $F$、终止符集 $T$ 与深度上限，所有合法表达式构成搜索空间
            $\mathcal{S}(F,T,D)$。目标 $f$ 可达当且仅当 $\exists T\in\mathcal{S},\ T(x)=f(x)$
            在讨论的定义域上成立。

            **表达力 vs 可搜索性**：这是两个独立问题。

            - $f$ 可表达，不代表有限预算内能搜到（搜索效率问题）；
            - 有限样本上拟合得好，不代表 $f$ 可精确表达（近似问题）。

            例：$\sin(x)$ **不是有理函数**（数学事实），而 $\{+,-,*,/\}$ 与 $x$ 生成的表达式都是有理
            函数，因此不存在能精确表示 $\sin$ 的有限表达式树。可是 GP 仍然可能在 20 个训练点上
            得到很小的 MSE——那是数值近似，不是“恢复了 sin 的公式”。

            **常数表示困难**：本项目终止符只有 $x$，没有随机常数；$1=x/x$、$0=x-x$、$1/2=1/(1+1)$
            等只能靠结构合成。系数搜索因此变成结构搜索的一部分，代价更高。

            **保护除法与数值语义**：

            $$p\_div(a,b)=\begin{cases}1.0,& b=0\\ a/b,& b\neq0\end{cases}$$

            - 分母精确为 0：直接返回 1.0，**改变**了数学语义；
            - 分母接近 0：做普通除法，可能得到极大值，放大数值误差；
            - 深树/极端输入可能溢出为 `inf`；`fitness` 对非有限预测返回 `inf`（明确策略）。

            因此标准代数化简（如 $\frac{x^2-1}{x-1}=x+1$）在保护除法下**不是全域等价**：
            在 $x=1$ 处左边按保护除法得 1.0，右边是 2。

            ### 具体小例子

            `(/ (- (* x x) (/ x x)) (- x (/ x x)))` 在 $x=1$ 处：分子 $1-1=0$，分母 $1-1=0$，
            保护除法返回 1.0；而 $x+1=2$。同一条“代数恒等式”在 $x\neq1$ 时成立，在 $x=1$ 处失效。

            ### 源码入口

            - 保护除法与求值：`gptoy/operators.py` → `p_div`、`evaluate`、`fitness`
            - 自定义目标数据：`gptoy/data.py` → `make_target_data`
            - 用自定义数据训练：`run_gp(config, data=...)`
            - 非有限值策略：`gptoy/operators.py` → `NON_FINITE_FITNESS`（`inf`）
        """),
        md(r"""
            ### E10：空间不匹配与数值语义

            **研究问题**：

            - A：目标含 $\sin$ 而算子集只有四则运算时，GP 会得到什么？低训练 MSE 能否证明结构恢复？
            - B：保护除法在除零点、近零分母与溢出时分别表现如何？哪些代数化简会失效？

            **核心概念**：可达性、算子集限制、表达力 vs 可搜索性、常数表示、保护除法、数值不稳定。

            **可修改配置**（改下面单元顶部变量后重跑）：

            - `E10_SIN_CFG` 的种群/代数/深度；
            - `E10_PROBES`：数值语义检查的输入点（含 0、近零、正常值）。

            **运行入口**：接下来两个代码单元（`# E10-A`、`# E10-B`）。

            **源码位置**：`make_target_data`、`p_div`、`fitness`。

            > **我的预测（请填写）**：四则运算 GP 在 sin 的 20 个训练点上能到多小的 MSE？
            > 它的测试 MSE 会小吗？（提示：sin 在 $[1,2]$ 上的形状与 $[-1,1]$ 很不同。）它学到的
            > 表达式大致是什么样子？保护除法在 $x=0$ 与 $x=10^{-320}$ 处分别会返回什么？
        """),
        code(r"""
            # E10-A 目标含 sin，但算子集只有 {+,-,*,/} 与 x
            E10_SIN_CFG = replace(cfg, seed=0, population_size=40, max_generations=15,
                                  max_tree_depth=9, stop_on_target=False)
            data_sin = make_target_data(np.sin, n_train=20, train_range=(-1.0, 1.0),
                                        n_test=10, test_range=(1.0, 2.0))
            e10_sin = run_gp(E10_SIN_CFG, data=data_sin)

            print_table(
                ["指标", "数值"],
                [
                    ("训练 MSE（sin, 20 点）", f"{e10_sin.train_mse:.4e}"),
                    ("测试 MSE（sin, [1,2] 10 点）", f"{e10_sin.test_mse:.4e}"),
                    ("best 表达式", e10_sin.best_of_run.sexpr_str()),
                    ("size / depth", f"{e10_sin.best_of_run.size()} / {e10_sin.best_of_run.depth()}"),
                    ("evaluations", e10_sin.evaluations),
                ],
            )
            print()
            print("sin 不是有理函数，因此不存在能精确表示它的有限四则运算表达式；")
            print("上面的训练 MSE 无论多小，都只能是有限样本上的数值近似。")
        """),
        code(r"""
            # E10-A 可视化：学到的函数 vs sin（注意纵轴范围被截断以便观察）
            xs_dense = np.linspace(-2.0, 2.0, 400)
            learned = evaluate(e10_sin.best_of_run, xs_dense)

            fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))
            axes[0].plot(xs_dense, learned, label="learned tree (four ops)")
            axes[0].plot(xs_dense, np.sin(xs_dense), "--", lw=1, label="true sin(x)")
            axes[0].plot(data_sin[0], data_sin[1], "o", ms=4, label="training points")
            axes[0].plot(data_sin[2], data_sin[3], "s", ms=4, label="test points")
            axes[0].axvline(1.0, color="gray", ls=":", lw=1)
            axes[0].set_ylim(-2.0, 2.0)
            axes[0].set_xlabel("x")
            axes[0].set_ylabel("y (clipped to [-2, 2])")
            axes[0].set_title("Four-op GP approximating sin")
            axes[0].legend(fontsize=7)

            axes[1].semilogy(e10_sin.best_mse_history, "-o", ms=3)
            axes[1].axhline(E10_SIN_CFG.mse_target, color="tab:red", ls="--", lw=1)
            axes[1].set_xlabel("generation")
            axes[1].set_ylabel("best train MSE (log)")
            axes[1].set_title("Training progress on sin")
            fig.tight_layout()

            print(f"在 [-2,2] 上 |learned - sin| 的中位数 = {np.median(np.abs(learned - np.sin(xs_dense))):.4g}")
            print(f"训练区间 [-1,1] 上的最大绝对误差 = {np.max(np.abs(evaluate(e10_sin.best_of_run, np.linspace(-1,1,200)) - np.sin(np.linspace(-1,1,200)))):.4g}")
        """),
        code(r"""
            # E10-B 保护除法的数值语义：除零点、近零分母、溢出
            E10_PROBES = (0.0, 1e-320, 1e-12, 1.0, 2.0)
            expressions = {
                "x/x": "(/ x x)",
                "(x-x)/x": "(/ (- x x) x)",
                "1/x（由 (x/x)/x 合成）": "(/ (/ x x) x)",
                "(x^2-1)/(x-1)": "(/ (- (* x x) (/ x x)) (- x (/ x x)))",
            }
            rows = []
            for name, sexpr in expressions.items():
                tree = str_to_tree(sexpr, cfg)
                values = [evaluate(tree, v) for v in E10_PROBES]
                rows.append((name, *[f"{v:.6g}" for v in values]))
            print_table(["表达式"] + [f"x={v:g}" for v in E10_PROBES], rows)
            print()
            ratio_tree = str_to_tree(expressions["(x^2-1)/(x-1)"], cfg)
            print("代数恒等式 (x^2-1)/(x-1) = x+1 的检验：")
            print_table(
                ["x", "保护除法结果", "x+1", "是否相等"],
                [(f"{v:g}", f"{evaluate(ratio_tree, v):.6g}", f"{v + 1:.6g}",
                  "是" if abs(evaluate(ratio_tree, v) - (v + 1)) < 1e-9 else "**否**")
                 for v in (0.5, 1.0, 2.0)],
            )
            print()
            overflow_tree = str_to_tree("(/ (/ x x) x)", cfg)
            probe = np.array([1e-320])
            print(f"溢出检查：evaluate 在 x=1e-320 处 = {evaluate(overflow_tree, 1e-320)}")
            print(f"fitness 对非有限预测的策略：{fitness(overflow_tree, probe, np.array([0.0]))}（NON_FINITE_FITNESS = inf）")
            print("说明：非有限值被判为最差个体，不会以 NaN 形式污染选择比较。")
        """),
        md(r"""
            ### 观察与解释（请填写）

            > **观察与解释（请填写）**：sin 实验的训练/测试 MSE 差多少？表达式是否呈现出对训练区间的
            > 近似而非 sin 的结构？（你无法从表达式直接看出 sin，因为 sin 不可达。）
            > 保护除法在 $x=0$ 与 $x=10^{-320}$ 处分别返回什么？哪一条代数恒等式失效？
            > 如果把非有限值策略改成“返回一个很大的有限数”，选择行为可能怎样变化？请写下猜测
            > 并标注为猜测。

            **本章诚实性说明**：

            - “sin 不可用有限四则运算精确表示”是数学事实（sin 不是有理函数），不是本次实验的观察；
            - 训练 MSE 小只说明 20 个点上的数值近似，不构成结构恢复；
            - 非有限值策略是**有意加入**的显式策略：默认 seed=3 的 Koza-1 轨迹经测试未受影响，
              只有在出现溢出/NaN 的病态个体时才会改变行为（见 README）。

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
# Chapter 11
# ---------------------------------------------------------------------------
def chapter_11() -> list:
    return [
        md(r"""
            ## Chapter 11：GP 失败机制与改进方法

            > 第三篇导言：本部分把 Chapter 0–10 的观察组织成可检验的失败机制与改进方向。

            **核心问题**：如何把观察到的失败现象组织成可检验的机制解释？前人可能的改进分别针对哪些机制？

            本章不引入新算法，而是把 Chapter 0–10 的观察整理成一张**失败分类表**。每一行都包含：
            机制假设、可观察指标、支持/反对证据、尚不能区分的替代解释、可调研的改进方向。
            请把它当作“研究问题生成器”，而不是结论清单。

            ### 概念与数学定义

            **失败机制**：一个可检验的因果陈述，说明“在什么条件下、哪个环节、通过什么过程”导致
            可观察的失败现象。它必须能推出**可区分的预测**，例如“若多样性耗尽是主因，则在相同预算下
            降低选择压力应提高达标率；若预算不足是主因，则降低选择压力收益有限而增加预算收益显著”。

            **失败分类的评价标准**：

            - 机制是否明确到可以控制变量？
            - 指标是否可观察、可复现（与种子/配置绑定）？
            - 证据是“观察”“对照实验”还是“猜测”？
            - 是否存在同样能解释现象的替代机制？

            ### 具体小例子

            本次 Notebook 的一个真实模式：小预算（E07，5 代）下有些种子训练 MSE 停在 $10^{-2}$ 量级，
            同时（E08-A）不同 k 的多样性曲线不同。仅凭这两点还不能断言“选择压力导致失败”，因为
            “预算不足”也能解释停滞。可检验的做法是：固定预算、只改 k，比较达标率；再固定 k、只改预算，
            比较达标率。**两种变化都有效，才说明两个机制都在起作用**（这仍是本次样本内的结论）。

            ### 失败分类表（7 类）

            | 失败类型 | 可能的机制解释 | 可观察指标 | 本次实验中的支持/反对证据 | 尚不能区分的替代解释 | 可调研的改进方向 |
            |---|---|---|---|---|---|
            | 1. 搜索效率不足 | 预算内触及不到可达的好结构；初始化/算子覆盖不足 | 首达代、评估次数、失败种子的 best MSE 平台 | E07：预算 5→15 时训练达标率变化；仍有失败种子 | 预算仍太小 / 搜索空间结构本身难达 / 选择效率低 | 更大预算、更强初始化、局部搜索（如常数优化）、并行多种子 |
            | 2. 多样性丧失与搜索停滞 | 选择压力过强、精英/复制过多，种群结构趋同 | `unique_ratio`、`fitness_std`、best MSE 平台长度 | E08-A：不同 k 的多样性曲线可比较；是否停滞需看曲线 | 收敛到正确解也会降多样性 / 种群规模太小 / 变异破坏性太强 | 小生境、物种形成、自适应变异率、年龄分层 |
            | 3. 表达式复杂度膨胀 | 选择间接奖励规模；中性/寄生结构累积 | `mean_size`、`max_size` 随代增长 vs best MSE 改善 | E08-A/B：规模曲线与原始 MSE 可对照；本实验未必出现膨胀 | 预算不足导致未及膨胀 / 深度上限截断 / 目标本身需要更复杂结构 | 简约压力、尺寸上限、多目标（Pareto）、结构重用惩罚 |
            | 4. 噪声与过拟合 | 目标利用训练噪声；小样本欠定 | 训练 vs 验证/测试 MSE 差、最终复杂度 | E09：不同噪声/样本量下的 train/val/clean-test 对照 | 可能是欠拟合而非过拟合 / 小预算导致两者都差 / 噪声尺度定义影响 | 正则化、验证集早停、数据增强、集成/多目标 |
            | 5. 表达式空间限制 | 函数集/终止符集无法表达目标；深度上限截断 | 训练 MSE 平台、表达式结构检查、目标是否可达的数学论证 | E10-A：sin 不可达，训练 MSE 再低也只是近似 | 也许是可表达但难搜索 / 采样点太少 / 保护运算改变语义 | 扩充算子集（sin/exp/log）、加入常数、元语法/强类型 GP |
            | 6. 常数优化困难 | 无随机常数；系数只能结构合成，搜索代价高 | 需要常数系数的目标上的收敛速度、表达式冗长程度 | Chapter 0：目标需要合成 1；E07/E08 的表达式常含 `(/ x x)` 等常数构造 | 也可能是算子集过窄 / 选择压力不足 | 常数注册与随机扰动、最小二乘系数优化、两阶段搜索 |
            | 7. 评价指标与模型选择问题 | 用训练/测试 MSE 代表“恢复”；用测试集调参或挑种子 | 报告口径、种子集合、是否只看最好种子、结构恢复率是否缺失 | E03：0 训练误差的多个结构；E07：训练/测试达标率分离；Chapter 0：数值一致但结构不同 | 也许 MSE 已足以服务预测任务（若目标只是预测） | 结构判别/等价类检验、独立验证集选择、预注册种子与指标、报告全部种子 |

            ### 如何区分“观察 / 机制 / 猜测 / 文献”

            | 层级 | 写法示例 | 需要什么证据 |
            |---|---|---|
            | 观察 | “预算 5 代时 6 个种子中 2 个训练达标” | 运行记录、种子、配置 |
            | 机制解释 | “关闭提前停止后仍失败，说明不是提前停止造成的” | 至少一个控制了关键变量的对照实验 |
            | 猜测 | “我猜是种群多样性太低导致” | 尚未检验；必须标注猜测并写出可检验预测 |
            | 数学事实/文献 | “sin 不是有理函数，所以不可精确表示” | 数学证明或可引用来源 |

            ### 研究问题模板

            一个可检验的问题应包含：**现象** + **机制候选** + **可区分预测**。例如：

            > 现象：小预算下失败种子的 best MSE 停在 $10^{-2}$ 附近。
            > 机制候选 A：预算不足；B：选择压力过强导致多样性耗尽。
            > 可区分预测：若 A 成立，增加预算应显著提高达标率且多样性下降更慢；若 B 成立，在相同预算下
            > 降低 k 应提高达标率，而单纯增加预算收益有限。→ 用 E07/E08 的组合实验检验。

            ### 源码入口

            本章不需要新源码；请回看 `gptoy/metrics.py`（可观察指标）、`gptoy/tracing.py`（逐代证据）、
            `gptoy/experiments.py`（对照实验与保存）以及各章源码入口。
        """),
        md(r"""
            ### 综合复盘：把前面实验的真实数字拼起来

            **研究问题**：在同一份 Notebook 运行中，各章实验的结果能否支持或反驳失败分类表中的某些行？
            哪些结论只是本次运行的观察，哪些还缺对照？

            **运行入口**：下一代码单元（`# E11`）只读取前面实验的真实结果对象，不重新运行算法、
            不硬编码任何数字。

            > **我的预测（请填写）**：把 E07（预算）、E08（选择压力/复杂度）、E09（噪声/样本）、
            > E10（空间）的结果放在一起，你认为本次运行最主要的失败类型是哪一类？你准备用哪个
            > 可区分预测去检验它？
        """),
        code(r"""
            # E11 综合复盘：读取前面实验的真实结果，生成一张“证据卡”
            evidence_rows = []
            for budget, results in e07_results.items():
                summary = summarize_runs(results)
                evidence_rows.append((f"E07 budget={budget}", f"train={summary['train_success_rate']:.2f}",
                                      f"test={summary['test_success_rate']:.2f}",
                                      f"hits={summary['first_hit_generation']['values']}"))
            for k, runs in e08a.items():
                last = [r.snapshots[-1] for r in runs]
                evidence_rows.append((f"E08-A k={k}", f"train={np.mean([r.train_mse for r in runs]):.3g}",
                                      f"unique={np.mean([s.unique_ratio for s in last]):.2f}",
                                      f"mean_size={np.mean([s.mean_size for s in last]):.1f}"))
            for name, r in e08b.items():
                evidence_rows.append((f"E08-B {name}", f"train={r.train_mse:.3g}",
                                      f"size={r.best_of_run.size()}",
                                      f"mean_size(last)={r.snapshots[-1].mean_size:.1f}"))
            for n_train, noise, subset in conditions:
                evidence_rows.append((f"E09 n={n_train},noise={noise}",
                                      f"train={np.mean([r['train'] for r in subset]):.3g}",
                                      f"val={np.mean([r['val'] for r in subset]):.3g}",
                                      f"clean_test={np.mean([r['test'] for r in subset]):.3g}"))
            evidence_rows.append(("E10-A sin", f"train={e10_sin.train_mse:.3g}",
                                  f"test={e10_sin.test_mse:.3g}",
                                  f"size={e10_sin.best_of_run.size()}"))
            print_table(["实验/条件", "指标 1", "指标 2", "指标 3"], evidence_rows)
        """),
        code(r"""
            # E11 失败诊断卡：挑一个真实失败种子（按训练 MSE，不按测试挑选），打印可观察指标
            worst_run = max(e07_results[min(E07_BUDGETS)], key=lambda r: r.train_mse)
            last = worst_run.snapshots[-1]
            print("失败诊断卡（来自 E07 最小预算组的训练最差种子）：")
            print_table(
                ["字段", "值"],
                [
                    ("seed", worst_run.seed),
                    ("stop_reason", worst_run.stop_reason),
                    ("generations / evaluations", f"{worst_run.n_generations} / {worst_run.evaluations}"),
                    ("final train MSE", f"{worst_run.train_mse:.4e}"),
                    ("final test MSE", f"{worst_run.test_mse:.4e}"),
                    ("best 表达式", worst_run.best_of_run.sexpr_str()),
                    ("size / depth", f"{worst_run.best_of_run.size()} / {worst_run.best_of_run.depth()}"),
                    ("last unique_ratio", f"{last.unique_ratio:.3f}"),
                    ("last fitness_std", f"{last.fitness_std:.4g}"),
                    ("mean_size / max_size", f"{last.mean_size:.1f} / {last.max_size}"),
                    ("best train MSE 轨迹", [f"{v:.3g}" for v in worst_run.best_mse_history]),
                ],
            )
            print()
            print("这不是自动结论。请回答：这张卡能支持失败分类表中的哪几行？")
            print("要区分这些行，你下一步会控制哪个变量、预期看到什么差异？")
        """),
        md(r"""
            ### 观察与解释（请填写）

            > **观察与解释（请填写）**：综合表里哪些数字支持“搜索效率不足”？哪些支持“多样性丧失”？
            > 哪些行在本次运行中**没有**得到支持（例如没有观察到膨胀）？失败诊断卡里最可疑的指标是哪个？
            > 请写出一个包含“现象 + 机制候选 + 可区分预测”的研究问题，并说明你还需要补哪个实验。

            ### 解释与局限

            - 这张分类表是**启发式组织方式**，不是互斥的完备分类；同一现象常有多类机制同时作用；
            - 表中的“支持证据”只来自本 Notebook 的小规模实验，不能当作普遍结论；
            - 未实现神经 GP/现代符号回归框架，因此“改进方向”只到可调研的问题层面，不代表已验证有效；
            - 外部文献结论在本章没有引用具体论文；若你补充文献，请注明来源并与本次观察分开记录。

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
# 附录
# ---------------------------------------------------------------------------
def appendix() -> list:
    return [
        md(r"""
            ## 附录 A：保存实验结果、放大规模与保存笔记

            ### A.1 保存可独立读取的实验结果

            下面的代码把 E07 的最小预算组保存为 JSON（配置、种子、全部结果、摘要）。文件写在
            `results/` 下，该目录被 `.gitignore` 忽略，不会污染代码仓库。

            > 注意：这类生成物只保存很小、可复现的聚合结果；不要提交大型输出。
        """),
        code(r"""
            # 保存 E07 小预算组：配置 + 种子 + 全部结果 + 摘要
            from pathlib import Path

            e07_small_budget = min(E07_BUDGETS)
            e07_small_summary = summarize_runs(e07_results[e07_small_budget])
            e07_save_path = save_experiment(
                Path(ROOT) / "results" / "lab_e07_small_budget.json",
                e07_base, e07_results[e07_small_budget], e07_small_summary,
            )
            payload = load_experiment(e07_save_path)
            print(f"已写入：{e07_save_path.resolve()}")
            print(f"文件大小：{e07_save_path.stat().st_size} bytes")
            print(f"顶层键：{sorted(payload)}")
            print(f"种子：{payload['seeds']}")
            print(f"结果条数：{len(payload['results'])}；摘要 n_runs：{payload['summary']['n_runs']}")
        """),
        md(r"""
            ### A.2 放大实验规模

            默认配置刻意很小，便于几十秒内跑完全部实验。若要得到更稳定的统计结论，请按下面的方式放大，
            并注意“先声明种子与指标，再看结果”：

            | 目标 | 建议改动 | 代价与注意 |
            |---|---|---|
            | 复现 Koza-1 成功案例 | 直接使用 `GPConfig(seed=3)`（种群 100、50 代） | 单次约 1 秒；seed=3 是旧版搜索出的教学种子 |
            | 多种子成功率 | `E07_SEEDS = range(20)`、`E07_BUDGETS = (10, 25, 50)` | 运行时间线性增长；仍只报告全部种子 |
            | 选择压力对照 | `E08A_KS = (1, 3, 7, 15)`、`E08A_SEEDS = range(5)` | 每个 k×seed 一次完整运行 |
            | 复杂度/噪声对照 | 增加 `E08B_LAMBDAS`、`E09_GP_SEEDS` | 简约惩罚 λ 越大，原始 MSE 通常越差，需同时报告两者 |
            | 更大搜索空间 | `GPConfig(max_tree_depth=17, population_size=100, max_generations=50)` | 深树可能极慢（膨胀），务必先小预算试跑 |

            放大时请保持：**测试集不参与选择**；**先固定种子集合**；**报告全部结果与失败案例**；
            **把结构恢复与数值拟合分开**。

            ### A.3 保存个人笔记

            - 直接在 Notebook 的“本章笔记”单元里填写 Markdown，然后 `Ctrl+S` 保存 `.ipynb`；
            - 也可以把结论同步到 `notes/GP_learning_notes.md`（与 Notebook 章节一一对应）；
            - **注意**：`scripts/build_learning_lab.py` 会重新生成 Notebook 与笔记骨架。为避免覆盖
              已填写的个人内容，脚本在目标文件已存在时默认**拒绝写入**；只有在明确传入 `--force`
              时才会覆盖。请先备份你的笔记。

            ### A.4 常见问题

            | 问题 | 处理 |
            |---|---|
            | `ModuleNotFoundError: gptoy` | 确认 kernel 为 `gptoy`，或在仓库根目录启动 Jupyter；第一个代码单元会把仓库根加入 `sys.path` |
            | 图不显示 | 确认已运行第一个代码单元的 `%matplotlib inline`；无头执行时设置 `MPLBACKEND=Agg` |
            | 运行很慢 / 内存升高 | 深树膨胀会使求值代价急剧上升；先降低 `max_tree_depth` 或 `max_generations` |
            | 结果与上次不同 | 检查 `seed` 是否固定；`data_seed` 与 GP `seed` 是两个独立随机源 |
            | 想清空输出重新运行 | Jupyter: Kernel → Restart & Run All |
        """),
    ]
