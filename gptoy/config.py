"""全局配置：``GPConfig`` 数据类与默认超参数（对齐 Success Criteria 3.3 的预算）。

所有可调参数集中在此，notebook 与 scripts 共享同一份默认值；
种子搜索（``scripts/find_seed.py``）通过构造新的 ``GPConfig`` 实例来调参。
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class GPConfig:
    # ---- 数据（3.3）----
    train_points: int = 20        # 训练集：[-1, 1] 等距 20 点
    train_lo: float = -1.0
    train_hi: float = 1.0
    test_points: int = 10         # 泛化集：[1, 2] 等距 10 点
    test_lo: float = 1.0
    test_hi: float = 2.0

    # ---- 函数集与终止符集（3.2：函数集 {+, -, *, /}，终止符集 {x}）----
    function_set: tuple = ("+", "-", "*", "/")
    terminal_set: tuple = ("x",)

    # ---- 预算（3.3：种群 100，最多 50 代）----
    population_size: int = 100
    max_generations: int = 50

    # ---- 树结构 ----
    min_init_depth: int = 2
    max_init_depth: int = 6
    max_tree_depth: int = 17      # 交叉/变异后的深度上限（Koza 经典值）
    variation_retries: int = 20   # 交叉超深时重新选交叉点的最大重试次数

    # ---- 遗传算子概率（三者之和 ≤ 1；reproduction = 余量）----
    # 默认配比保证选择/交叉/变异在运行中均可观测（3.1）
    p_crossover: float = 0.7
    p_mutation: float = 0.2
    tournament_size: int = 7
    elitism_size: int = 1

    # ---- 判定阈值（3.1 / 3.3）----
    mse_target: float = 1e-6           # 训练/泛化 MSE 达标线
    gen0_unsolved_mse: float = 1e-3    # 第 0 代 best MSE 必须高于此值（未在第 0 代解出）
    # 训练 MSE 达标后是否立即停止；False 时继续跑满 max_generations（长期行为实验用）。
    # 默认 True 与旧版 run_gp 的提前收敛行为完全一致。
    stop_on_target: bool = True

    # ---- 可复现性（3.3）----
    seed: int | None = None            # 由 find_seed.py 搜索后写入 notebook 固定值
