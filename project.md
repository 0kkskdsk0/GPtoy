# Project Brief

## 1. Goal
背景：本项目的研究目的是创建一个学习遗传编程的教学案例。不要求算法算子在搜索广度或计算速度上有多高级，只要求案例足够经典和完善，只要求能够让我这个只懂概念的基础者了解遗传编程的具体细节。因此该案例需要从零开始实现，鉴于语法简洁性，要求使用python实现，且不能调用现成的遗传编程库（常用的python库，如科学计算相关的库等可以使用）。计划主体教学文件为一个notebook，其展示了算法的执行步骤，并使用了markdownb cell和代码进行了必要说明。但notebook的每个步骤内的具体算子则要求在另外的python文件中实现，notebook只负责调用这些抽象算子，以保持执行的简明性。再次注意，这些具体算子要用python从零开始实现。
**经典求解问题选择**： Koza-1，使用GP算法从 20 个数据点里恢复 f(x)=x⁴+x³+x²+x

## 2. Deliverable
- 一个notebook文件，内含对一个经典问题使用抽象算子执行GP算法的完整求解步骤
- 多个python文件（按GP执行步骤分划），内含对每个阶段的抽象算子的从零实现
（注：实现时可按需创建其它文件，比如代码测试文件，但上述交付产物一定要存在）

## 3. Success Criteria

### 编写原则：
- 对定义项目成功所需要考虑的特性进行分门别类，每一个类别下面要强调**如何验证**（三要素：观察对象、判据和判定者；适当加入正负向案例），但不用考虑具体的验证步骤；

#### tips
- 要区分定义成功和写测试；验证器的存在是为了让人更好地做产物是否正确的判断，验证器应该比验证物简单；
- 对验证器的验证应该就是语义正确（人/LLM审查）+冒烟测试通过

### 声明：
本项目是教学案例，不是算法竞赛，不必追求搞性能。若需判定该项目是否成功完成，即需检查本节下面3.1~3.4对成功项目的定义

### 3.1 教学走得通
- Notebook 按gp主循环展开，每一步都有 markdown 说明；代码单元只调用外部算子，不内嵌该步的实现。
- 固定随机种子后，notebook 能在内核中从头到尾一次执行完毕，运行于10分钟内结束，成功找回目标表达式。
- 所固定的随机种子不能让算法在第 0 代就找到答案。否则后面的选择、交叉、变异没有可观察的进化过程。

### 3.2 实现是从零写的遗传编程
- 所有算子写在按阶段拆分的 Python 文件里，不调用任何现成遗传编程库。
- 个体是树。适应度必须通过对树递归求值得到，不能把目标公式写死在评价函数里。
- 初始种群由函数集 `{+, -, *, /}` 与终止符集 `{x}` 随机生成，不把目标表达式注入初始个体。

### 3.3 在约定预算内找回 Koza-1
- 目标表达式为 `f(x)=x^4+x^3+x^2+x`，训练集由该表达式在 `[-1, 1]` 上等距取20点得到。
- 预算：种群规模 100，最多 50 代。在此预算内，最好个体（best-of-run）在训练集上的均方误差小于 `1e-6`。
- 验证泛化性：另在`[1,2]`上等距取10点，要求最好个体在这组点上的均方误差同样小于 `1e-6`，避免只是插值了那 20 个训练点。
- 固定随机种子：遗传编程有随机性。成功不要求所有种子都找回，但必须存在一个写进 notebook 的种子，使上述标准可复现。

### 3.4 项目实现环境规定
- 该项目要求使用conda创建新的python 3.12环境予以实现，环境命名为“GPtoy”

## 4. Implementation Scheme

**总体思路**：以 3.1–3.4 为验收标准，按“环境 → 个体表示 → 初始化 → 评价 → 选择 → 变异 → 进化循环 → 种子搜索 → Notebook 编排 → 验证”的顺序落地。三条贯穿性原则：

1. **抽象边界**：所有算子按 GP 阶段拆分到 `gptoy` 包内独立模块，notebook 只做编排与讲解，代码单元只调用抽象算子（对齐 3.1）；
2. **可复现**：全项目随机性统一由 `utils.make_rng(seed)` 注入单一 `random.Random` 实例，同一种子运行结果逐位一致（对齐 3.3）；也就是只要种子一致，算法的行为就完全确定，包括随机行为，这使得结果可排除随机性复现
3. **从零实现**：不引入任何现成遗传编程库（deap / gplearn 等），仅使用科学计算常用库 numpy 与绘图库 matplotlib（对齐 3.2）。

### 阶段 0：环境与骨架（对齐 3.4）
- 用 conda 创建 python 3.12 环境 `GPtoy`（`environment.yml` 声明），依赖：numpy、matplotlib、jupyterlab、ipykernel；开发/验证依赖：pytest、nbclient、nbformat。
- 建立包骨架：`gptoy/`（阶段拆分模块）、`tests/`（冒烟测试）、`scripts/`（种子搜索与验证器）、`notebooks/`（教学主体）；`pip install -e .` 后 notebook 可直接 `import gptoy`。
- **挑战**：环境名与 python 版本必须与 3.4 逐字一致；editable install 要保证 notebook 与脚本在任何工作目录下 import 路径稳定。

### 阶段 1：个体表示与算子（对齐 3.2 前半）
- `Node` 树：函数节点 `{+, -, *, /}`（arity 2）、终止符 `{x}`；提供深度/规模统计、中缀与 S-表达式显示、字符串解析（供测试与展示）。
- 保护除法：除数为 0 时返回 1.0（Koza 经典约定），杜绝 NaN/Inf 污染适应度。
- 递归求值 `evaluate(node, x)`：按树后序递归，x 支持标量与 ndarray——对 20 个训练点向量化，兼顾“适应度必须通过对树递归求值得到”（3.2）与 notebook 运行时长（3.1）。
- **挑战**：fitness 接口只接收 `(tree, x_train, y_train)`，目标公式绝不进入评价函数（3.2 红线）；float64 精度需足以分辨 1e-6 量级 MSE。

### 阶段 2：初始化（对齐 3.2 后半）
- Koza 经典 **ramped half-and-half**：对深度 2–6 的每个深度桶，一半 `full`（所有叶子在同一深度）一半 `grow`（形状可变），随机生成，**不注入目标表达式**。
- 可达性论证：搜索空间（仅 `{x}` 与四则运算）包含目标，因为 `x⁴+x³+x²+x = x·(x+1)·(x²+1)`，且 `x-x`、`x/x`（保护除法）可合成常数 0、1，故无需随机常数也能表达目标，种子搜索可行。
- **挑战**：终止符集比经典 Koza（含随机常数）更窄，多样性靠 ramped 初始化与交叉/变异维持；种群规模必须恰为 100。

### 阶段 3：评价与选择
- 适应度 = 训练集（`[-1,1]` 等距 20 点）上的 MSE；`best-of-run` 取历代最优；数据标签由 `data.py` 的目标公式生成（公式唯一合法出现位置）。
- 锦标赛选择（`k=7`，有放回）：实现简单、无需处理负适应度，notebook 中与适应度比例选择作对比讲解。
- **挑战**：锦标赛大小权衡（过小随机漂移、过大早熟收敛）；统一“MSE 越小越优”约定。

### 阶段 4：变异算子与进化循环
- subtree crossover：随机交换两亲本子树，深度超限则重试（有限次）后回退为亲本副本；subtree mutation：随机子树替换为 grow 新树；reproduction：深拷贝。
- 生成式（generational）主循环 + 精英保留 1 棵；概率配比（默认 0.7/0.2/0.1）保证选择、交叉、变异在运行中均可观测（3.1）；记录历代 best MSE 与首达代。
- **挑战**：交叉后深度爆炸是经典问题，深度上限 + 重试策略必须可靠；所有算子必须保持树结构合法（节点集合、arity、深度约束）。

### 阶段 5：种子搜索（对齐 3.1、3.3）
- 离线脚本 `scripts/find_seed.py` 在候选种子范围搜索，要求同时满足：① 第 0 代 best MSE > 1e-3（未在第 0 代解出，进化过程可观测）；② ≤ 50 代内训练 MSE < 1e-6；③ 泛化集 `[1,2]` 等距 10 点 MSE < 1e-6。找到的种子写入 `results/found_seed.json` 并在 notebook 中固定。
- **挑战**：全项目最不确定环节——需在小范围内调参（锦标赛大小、深度上限、算子概率）找到“既快又稳”的种子；泛化要求学到真实多项式而非插值；需留足运行余量确保 notebook 10 分钟内完成。

### 阶段 6：Notebook 教学编排与总验证（对齐 3.1）
- notebook 按 GP 主循环分 8 节：问题与数据 → 表示 → 初始化 → 评价 → 选择 → 交叉/变异 → 主循环 → 结果与泛化；每节 markdown 讲解 + 代码单元只调用 `gptoy.*` 抽象算子。
- 验证链：结构审查（每节有 markdown、代码单元无内嵌实现）→ pytest 冒烟测试 → nbclient 全量执行 notebook（断言无异常、计时 < 10 分钟、终值达标）→ `scripts/verify_project.py` 汇总 3.1–3.4 PASS/FAIL。
- **挑战**：教学叙事与抽象边界的平衡——算子命名贴近教科书术语，每节只暴露一个抽象操作。

### 方案 → 成功标准映射

| 成功标准 | 方案对应 | 判定方式 |
|---|---|---|
| 3.1 教学走得通 | 阶段 6 + 阶段 5 | 结构审查 + nbclient 全量执行 + gen-0 未解出断言 |
| 3.2 从零实现 | 阶段 1–4 | 依赖扫描（无 GP 库）+ 递归求值冒烟测试 + 节点集合断言 |
| 3.3 预算内找回 | 阶段 5 | find_seed / verify 断言训练与泛化 MSE < 1e-6 |
| 3.4 环境规定 | 阶段 0 | conda env list + python 版本检查 |

### 主要风险与对策
- 种子不可复现 → 全链路单一 RNG，禁止散落 `random`/`np.random` 裸调用；
- 深度爆炸 / 除零 NaN → 深度上限 + 重试、保护除法；
- 泛化失败（只插值不学公式）→ 提高初始化多样性、调整参数后重新搜种子；
- notebook 超时 → 向量化递归求值 + 收敛提前终止。

## 5. Project Architecture

### 目录结构（当前为骨架/占位，待按第 4 节方案实现）

```
GPtoy/
├── project.md                # 项目说明（本文件）
├── README.md                 # 使用说明：环境创建、安装、运行、验证
├── environment.yml           # conda 环境定义（name: GPtoy, python 3.12）
├── requirements.txt          # 运行时依赖（numpy, matplotlib）
├── requirements-dev.txt      # 开发/验证依赖（pytest, nbclient, nbformat, jupyterlab）
├── pyproject.toml            # 包定义（pip install -e . 后 notebook 可 import gptoy）
├── .gitignore
├── gptoy/                    # 从零实现的 GP 算子包（按阶段拆分）
│   ├── __init__.py           # 包入口，导入并 re-export 各阶段模块
│   ├── config.py             # GPConfig 数据类 + 默认超参（预算/概率/深度/阈值）
│   ├── data.py               # Koza-1 目标公式与训练/泛化数据集（公式唯一合法出现处）
│   ├── representation.py     # Node 树：构造、depth/size、中缀与 S-表达式、str_to_tree
│   ├── operators.py          # 保护四则运算、递归 evaluate、MSE fitness
│   ├── initialization.py     # ramped half-and-half（full / grow / 初始种群）
│   ├── selection.py          # 锦标赛选择
│   ├── variation.py          # subtree crossover / subtree mutation / reproduction
│   ├── evolution.py          # run_gp 生成式主循环、精英保留、RunResult
│   └── utils.py              # 单一 RNG（make_rng）、全局种子防御
├── notebooks/
│   └── koza1.ipynb           # 教学主体：按主循环分 8 节，只调用 gptoy 抽象算子
├── tests/                    # 冒烟测试（每个模块一个文件，验证语义正确性）
│   ├── conftest.py           # sys.path 引导，保证未安装时也可 import gptoy
│   ├── test_representation.py
│   ├── test_operators.py
│   ├── test_initialization.py
│   ├── test_selection.py
│   ├── test_variation.py
│   └── test_evolution.py     # 对 3.1/3.3 的直接验证（种子收敛、gen-0 未解出、可复现）
├── scripts/
│   ├── find_seed.py          # 离线种子搜索（3.1/3.3 前置），结果写 results/found_seed.json
│   └── verify_project.py     # 成功标准自动化验证器（3.1–3.4 PASS/FAIL，非零退出码）
└── results/                  # 生成物：found_seed.json、收敛曲线图等（git 忽略）
```

### 模块依赖方向（自上而下单向，无环）

```
notebooks/koza1.ipynb
        │  import
        ▼
   gptoy.evolution ──► gptoy.initialization ──► gptoy.representation ◄──┐
        │                gptoy.selection ──► gptoy.operators ───────────┤
        │                gptoy.variation ───► gptoy.operators ──────────┤
        ▼                                        ▲                       │
   gptoy.data ◄──────（fitness 只接收 x, y，不读公式）                     │
   gptoy.config ◄────────────────────────────────────────────────────────┘
   gptoy.utils（make_rng，被所有随机模块使用）
```

- `config` / `utils` / `data` 为最底层；`representation` 依赖 `config`（节点集合定义）；
- `operators`（evaluate/fitness）依赖 `representation`，数据以参数传入，目标公式不进入评价函数；
- `initialization` 依赖 `representation`（构造合法树）；`selection` 依赖 `operators`（按 MSE 比较）；`variation` 依赖 `representation` 与 `initialization`（生成替换子树）；
- `evolution` 依赖以上全部，对外只暴露 `run_gp(config) -> RunResult`；
- `tests` / `scripts` 依赖 `gptoy` 包；notebook 依赖 `gptoy` 与 matplotlib（绘图）。

### 执行数据流

```
GPConfig ──► make_rng(seed)                      # 单一随机源，保证可复现
     │
     ▼
data.py：x_train(20 点 @[-1,1])、y_train=f(x_train)；x_test(10 点 @[1,2])、y_test
     │
     ▼
initialization：ramped half-and-half → P0（100 棵 {x, +,-,*,/} 树）
     │
     ▼
evolution 主循环（≤ 50 代）：
   fitness(Pi) = MSE(evaluate(tree, x_train), y_train)     # 树递归求值
   → 记录 best-of-run → 精英保留 1 棵
   → 锦标赛选择父本 → crossover / mutation / reproduction 生成 Pi+1
     │
     ▼
RunResult{best_of_run, best_mse_history, generation_found, train_mse,
          test_mse, gen0_mse, n_generations, total_seconds}
     │
     ▼
notebook 展示：收敛曲线、最终表达式、训练/泛化 MSE 对比
scripts/verify_project.py → 3.1–3.4 PASS/FAIL
```

### Notebook 分节与模块映射

| Notebook 节 | 讲解内容 | 调用的抽象算子 |
|---|---|---|
| 1 问题与数据 | Koza-1 定义、训练/泛化集 | `gptoy.data.make_koza1_data` |
| 2 个体表示 | 树、节点、显示 | `gptoy.representation` |
| 3 初始化 | ramped half-and-half | `gptoy.initialization.initial_population` |
| 4 评价 | 保护运算、递归求值、MSE | `gptoy.operators.evaluate / fitness` |
| 5 选择 | 锦标赛选择 | `gptoy.selection.tournament_select` |
| 6 交叉与变异 | 子树交叉/变异/复制 | `gptoy.variation.*` |
| 7 主循环 | 生成式进化、精英保留 | `gptoy.evolution.run_gp` |
| 8 结果与泛化 | 收敛曲线、泛化验证 | `gptoy.utils` + matplotlib |

### 关键设计决策（与成功标准的对应）
1. 单一 `random.Random(seed)` 贯穿所有随机操作 → 可复现（3.3）；
2. 目标公式只存在于 `data.py`，fitness 只吃 `(tree, x, y)` → 不违反 3.2；
3. 模块按 GP 阶段命名（representation / initialization / selection / variation / evolution）→ notebook 每节对应一个模块，教学映射清晰（3.1）；
4. 测试遵循“验证器应比验证物简单”：每个模块一个冒烟测试文件，只验证语义（保护除法、深度约束、确定性、种子收敛），不做覆盖率竞赛。
