# GPtoy — 从零实现遗传编程的教学案例（Koza-1）

不使用任何现成遗传编程库，用 Python 从零实现经典问题 **Koza-1**：从 20 个数据点上恢复 `f(x) = x^4 + x^3 + x^2 + x`。主体教学文件为 notebook，算子按 GP 执行阶段拆分在 `gptoy` 包中。

> 设计细节与验收标准见 `project.md`（第 4 节为执行方案，第 5 节为项目架构）。

## 环境（对齐 Success Criteria 3.4）

```bash
conda env create -f environment.yml      # 创建 python 3.12 环境 GPtoy
conda activate GPtoy
pip install -e .                         # editable 安装，notebook/脚本可直接 import gptoy
```

> 环境坑位记录（本机已处理）：
> 1. `C:\Users\TYA\.condarc` 原含失效频道 `anaconda/pkgs/free`（404）与已被
>    清华镜像封锁的 `anaconda/pkgs/main`（403）——已改为可达的
>    `mirrors.sustech.edu.cn/anaconda/pkgs/main` + `defaults`（备份：
>    `C:\Users\TYA\.condarc.bak-20260819-205427`）。若换机器/换配置后仍有
>    失效镜像，创建环境时可用 `conda env create -f environment.yml --override-channels -c conda-forge` 绕过；
> 2. 注意：写 `.condarc` 必须用**无 BOM 的 UTF-8**（PowerShell `Set-Content`
>    默认带 BOM，会导致 conda 在 GBK 区域设置下静默忽略 `channels` 键）；
> 3. conda-forge 的 `numpy 2.5.x` / `matplotlib 3.11.1` win-64 构建会使多边形
>    填充渲染（`ax.hist`/`ax.bar`）原生崩溃，`environment.yml` 已把两者
>    分别固定在 `<2.5` 与 `<3.11`。

## 运行

```bash
jupyter lab notebooks/koza1.ipynb        # 教学主体（按 GP 主循环分 8 节，固定种子 SEED=3）
pytest tests/ -q                         # 冒烟测试（39 个用例）
python scripts/find_seed.py              # 离线种子搜索（预算内找回 + 第 0 代未解出）
python scripts/build_notebook.py         # 用 found_seed.json 的种子重新生成 notebook
python scripts/verify_project.py         # 成功标准 3.1–3.4 自动化验证（PASS/FAIL）
```

当前固定种子 `SEED = 3`：第 0 代 best MSE ≈ 0.0965（> 1e-3，进化可观测），
第 5 代收敛，训练/泛化 MSE ≈ 1e-32 / 3e-30，全程约 3 秒。

## 目录

```
GPtoy/
├── project.md          # 项目说明（目标 / 交付物 / 成功标准 / 方案 / 架构）
├── gptoy/              # 从零实现的 GP 算子包（data/representation/operators/...）
├── notebooks/          # 教学 notebook（koza1.ipynb，8 节）
├── tests/              # 冒烟测试（每模块一个文件）
├── scripts/            # find_seed.py（种子搜索）、build_notebook.py（notebook 生成）、
│                       # verify_project.py（验证器）
└── results/            # 生成物：found_seed.json、收敛曲线图等
```

## 状态

已完成：`gptoy` 全部算子从零实现并通过 39 个冒烟测试；种子搜索找到 `SEED=3`；
notebook 全量执行（nbclient）约 3 秒完成并找回目标表达式；`verify_project.py`
对 Success Criteria 3.1–3.4 全部 PASS。
