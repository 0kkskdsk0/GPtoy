# GPtoy — 从零实现的遗传编程学习实验室（Koza-1）

不使用任何现成遗传编程库，用 Python 从零实现经典树型遗传编程（GP），并把它扩展成一个
**教学优先、可复现、可长期保存**的学习实验室：

- `notebooks/GP_learning_lab.ipynb`：主教学 Notebook，Chapter 0–11 + 实验 E01–E10，真实运行；
- `notes/GP_learning_notes.md`：与 Notebook 章节/核心问题完全一致的可独立审阅笔记骨架；
- `notebooks/koza1.ipynb`：保留的原 Koza-1 教学案例（仅修正了 6.3 的父本数量注释，仍可执行）；
- `gptoy/`：算法与实验的 Python 实现（Notebook 只做编排、调用与可视化）。

> 设计背景与原始成功标准见 `project.md`。

## 环境

```bash
conda env create -f environment.yml      # python 3.12 环境 GPtoy
conda activate GPtoy
pip install -e .                         # 可选：editable 安装；不安装也可在仓库根目录运行
```

开发/验证依赖见 `requirements-dev.txt`（pytest、nbclient、nbformat、jupyterlab）。
Jupyter kernel 名为 **`gptoy`**（`GPtoy (Python 3.12)`）。无网络、无需下载数据。

## 快速开始

```bash
jupyter lab notebooks/GP_learning_lab.ipynb   # 主学习入口（kernel 选 gptoy）
jupyter lab notebooks/koza1.ipynb             # 原始教学案例（kernel 也要选 gptoy）
pytest -q                                      # 默认快速测试（不跑重型 Notebook）
pytest -q -m slow                              # 重型：Notebook 全量执行等
pytest -q -m "slow or not slow"                # 显式运行全部（含 slow）
```

无头执行 Notebook 时设置 `MPLBACKEND=Agg`（Notebook 第一个代码单元已 `%matplotlib inline`）。

## 学习路线

| 篇章 | 章节 | 实验 | 主题 |
|---|---|---|---|
| 第一篇：GP 的经典运行机制 | Chapter 0 | — | 符号回归、Koza-1 基线、数值拟合 vs 结构恢复 |
| | Chapter 1 | E01 | 表达式树、结构 vs 语义、保护除法 |
| | Chapter 2 | E02 | full / grow / ramped half-and-half |
| | Chapter 3 | E03 | MSE、欠定问题、训练/验证/测试职责 |
| | Chapter 4 | E04 | 锦标赛选择、选择压力、多样性 |
| | Chapter 5 | E05 | 子树交叉/变异/复制、深度回退 |
| | Chapter 6 | E06 | 逐代引擎、精英保留、Hall of Fame、收敛曲线 |
| 第二篇：GP 的典型失败 | Chapter 7 | E07 | 随机性、搜索预算、多种子分布 |
| | Chapter 8 | E08 | 早熟收敛、表达式膨胀、简约压力 |
| | Chapter 9 | E09 | 小样本、噪声、过拟合与泛化 |
| | Chapter 10 | E10 | 搜索空间限制、保护除法与数值语义 |
| 第三篇：从实验走向研究 | Chapter 11 | 综合 E01–E10 | 失败分类表与研究问题 |

每个实验都遵循：研究问题 → 预测 → 配置/控制变量 → 运行 → 观察 → 解释与局限。
个人预测/观察/总结区域全部留空（`我的预测（请填写）` / `待填写`），Notebook 不替你下结论。

## 目录

```
GPtoy/
├── gptoy/
│   ├── config.py         # GPConfig（含 stop_on_target）
│   ├── data.py           # Koza-1 数据 + make_target_data（自定义目标，如 sin）
│   ├── representation.py # Node：结构相等/哈希、路径访问、替换、S-表达式
│   ├── operators.py      # 保护四则运算、evaluate、fitness（非有限值 → inf）
│   ├── initialization.py # full / grow / ramped half-and-half
│   ├── selection.py      # 锦标赛选择（节点版 + 下标版）
│   ├── variation.py      # 交叉/变异/复制 + traced 接口（真实记录交叉点/变异点/回退）
│   ├── evolution.py      # EvolutionEngine 逐代引擎 + run_gp + RunResult
│   ├── metrics.py        # structural_signature、population_stats、complexity
│   ├── tracing.py        # GenerationSnapshot、GeneticEvent、EvolutionTracer/NullTracer
│   ├── experiments.py    # 多种子批量、数据划分/噪声、控制变量实验、JSON 保存
│   └── viz.py            # 表达式树绘制（高亮交叉点/变异点）
├── notebooks/
│   ├── GP_learning_lab.ipynb  # 主教学 Notebook（已真实执行并保存输出）
│   └── koza1.ipynb            # 原教学案例（保留，仅修正 6.3 注释）
├── notes/GP_learning_notes.md # 学习笔记骨架（12 章，个人区留空）
├── tests/                     # pytest（默认快速；slow 标记重型 Notebook 执行）
├── scripts/
│   ├── find_seed.py           # 原教学种子搜索（使用训练+测试筛选，见下）
│   ├── build_notebook.py      # 生成旧 koza1.ipynb
│   ├── verify_project.py      # 原项目成功标准验证器
│   └── build_learning_lab.py  # 生成学习实验室 Notebook 与笔记骨架（默认拒绝覆盖）
├── results/                   # 生成物（.gitignore 忽略；notebook 会把 E07 摘要写到这里）
└── environment.yml / requirements*.txt / pyproject.toml
```

## 算法与原有语义

- Koza-1：训练集 `[-1, 1]` 等距 20 点，测试/泛化集 `[1, 2]` 等距 10 点；目标
  `f(x)=x^4+x^3+x^2+x` 只用于生成标签，`fitness(tree, x, y)` 不读取公式；
- 函数集 `{+,-,*,/}`、终止符集 `{x}`；保护除法在除数为 0 时返回 1.0；深度上限 17；
- 锦标赛选择（有放回，MSE 小者胜）、子树交叉、子树变异、复制、精英保留、生成式主循环；
- 单一 `random.Random` 贯穿所有随机操作；固定 seed 逐位可复现；
- 默认配置下 `seed=3` 仍能在 50 代内训练/测试 MSE < 1e-6，第 0 代 best MSE > 1e-3
  （`tests/test_evolution.py` 用常量种子验证，不再依赖未跟踪的 `results/found_seed.json`）。

### 逐代引擎

`EvolutionEngine(config, tracer=None)` 是唯一算法实现：

```python
engine = EvolutionEngine(cfg, EvolutionTracer())
engine.initialize()
while not engine.done:
    engine.step()
result = engine.result
```

`run_gp(config, tracer=None)` 等价于 `EvolutionEngine(...).run()`；相同 seed/config 下，
逐代执行与一次性执行的 `best_mse_history`、最终表达式与 `evaluations` 完全一致。
`EvolutionTracer` 开启/关闭不改变 RNG 消费顺序，因此不改变搜索轨迹。

评估计数 `evaluations` 只统计**为搜索而做的训练适应度评估**（每评价一代计 `population_size`）；
测试集评价与诊断评价不计入预算，也从不反馈给选择。

## 数值异常策略（会影响原有行为吗？）

`operators.fitness` 对**非有限预测**（inf/nan）返回 `inf`，即把该个体判为最差；`p_div` 只把
**精确为 0** 的分母替换为 1.0，接近 0 的分母仍做普通除法。

- 默认 Koza-1 `seed=3` 的 50 代轨迹不触发该分支，搜索历史逐位不变（已有回归测试）；
- 旧实现若遇到 NaN，可能让 NaN 参与 `min` 比较；新策略把病态个体固定为 `inf`，行为更明确；
- 这是有意的行为修复，`README` 与 `notebooks` 中均有说明，并由测试覆盖。

## 复现性与种子

- GP 搜索随机性由 `GPConfig.seed` 控制（单一 `random.Random`）；
- 噪声/数据随机性由 `make_train_val_test(seed=...)` 控制（`numpy.random.default_rng`），
  与 GP 种子分离；
- **`seed=3` 的来源**：旧版 `scripts/find_seed.py` 在候选种子中先用“训练+测试同时达标”筛选，
  再按首达代排序挑出的教学种子。它展示的是被选中的成功样本，**不能**当作无偏泛化成功率；
  真正的成功率请看第 7 章的多种子统计（测试集不参与筛选）。
- `scripts/find_seed.py` 保留教学用途，其文档与输出也说明这一筛选性质。

## 实验输出与保存

- 默认实验规模很小（种群 20–100、代数 5–50、种子 2–6 个），完整 Notebook 约 20 秒；
- 放大方法见 Notebook **附录 A.2**（例如把 E07 的种子数改到 `range(20)`、预算加到 50）；
- `gptoy.experiments.save_experiment(path, config, results, summary)` 写出可独立读取的 JSON
  （配置、种子、全部结果、摘要），Notebook 附录 A.1 会写到 `results/lab_e07_small_budget.json`；
- `results/` 被 `.gitignore` 忽略，不会污染 `git status`；不要把大型输出提交进仓库。

## 学习笔记

- Notebook 每章末尾有 6 节笔记区（核心概念 / 实现机制与源码位置 / 我的实验问题与预测 /
  实验配置及观察结果 / 我的理解、替代解释与证据边界 / 本章总结与后续问题）；
- `notes/GP_learning_notes.md` 是同一套章节与核心问题的独立笔记骨架，个人填写区均为 `待填写`；
- 重新生成 Notebook/笔记：`python scripts/build_learning_lab.py --force`。该脚本在目标文件已存在时
  **默认拒绝覆盖**，以免抹掉已填写的个人笔记；请先备份。

## 测试

```bash
pytest -q            # 默认快速测试（算法/追踪/实验/教学交付物静态检查）
pytest -q -m slow    # 重型：nbclient 从头执行 GP_learning_lab.ipynb 并检查输出
pytest -q -m "slow or not slow"   # 显式运行全部（含 slow）
```

测试覆盖：

- Koza-1 `seed=3` 基线收敛、gen0 未解出、可复现、评估计数不变量；
- `evaluate` 标量/向量、保护除法零/近零、非有限值策略；
- full/grow/ramped 语义与确定性；锦标赛数量/压力/有放回/确定性；
- 交叉/变异子代合法、深度约束、回退、traced 与普通接口 RNG 一致；
- `Node.__eq__`/`__hash__` 一致、set 去重；
- `step()` 与 `run_gp()` 历史一致、tracer 开关不改变轨迹、事件真实对应、精英/回退记录；
- 数据划分不重叠、测试集不反馈、噪声公式可复现、多种子结果全部保留、JSON 保存可读取；
- 新 Notebook 可从头执行、E01–E10 都有真实输出、每章有 Markdown 与笔记占位；
- 旧 Notebook 仍可执行，6.3 父本数量注释已修正（20 个）。

## 已知限制

- **没有严格的结构恢复判定器**：`summarize_runs` 的结构恢复率默认是 `None`；只有显式传入已知
  目标结构（精确树匹配）时才会计算，且该比例是下界（等价但结构不同的树计为未恢复）；
- 测试 MSE 小只支持“数值拟合”，不构成“公式被严格恢复”的证明；
- 默认实验规模很小，统计结论只适用于本次观察，不应外推为普遍规律；
- 深度 17 的深树可能非常慢（膨胀），多种子实验请使用较小的 `max_tree_depth` 或先小预算试跑；
- 旧 `koza1.ipynb` 的元数据 kernelspec 仍是 `python3`（本机该 kernelspec 可能损坏），
  请显式选择 `gptoy` kernel 执行；
- 不包含神经 GP、现代符号回归框架或外部基准数据集。

## 验证命令（摘要）

```bash
pytest -q
pytest -q -m slow
python scripts/build_learning_lab.py --check
```
