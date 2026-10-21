"""项目自动化验证器（对应 ``project.md`` 3.1–3.4 + 学习实验室交付物）。

输出逐条 PASS/FAIL 并以非零退出码表示失败：

- ``[3.1]`` 旧教学 notebook 结构 + nbclient 全量执行 + 时长 + gen-0 未解出；
  固定种子直接取自 notebook 中的 ``SEED = N``，**不再依赖** ``results/found_seed.json``；
- ``[3.2]`` 依赖扫描（无 deap / gplearn 等 GP 库）+ fitness 只吃 (tree, x, y)
  + 初始种群节点集合；
- ``[3.3]`` 用 notebook 中固定的种子运行：训练/测试 MSE < 1e-6 且代数 ≤ 50；
- ``[3.4]`` python 版本 ≥ 3.12 且 conda 环境 ``GPtoy`` 存在；
- ``[lab]`` 学习实验室交付物：新 Notebook / 笔记骨架 / Chapter 0–11 / E01–E10 /
  笔记与 Notebook 章节一致 / 旧 notebook 6.3 注释修正。

用法（在仓库根目录、GPtoy 环境内执行）：
    python scripts/verify_project.py
"""

import os
import re
import subprocess
import sys
import time
from pathlib import Path

# Windows 控制台可能为 GBK：统一按 UTF-8 输出，避免中文/数学符号报错
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

# 无头执行 notebook 时强制 matplotlib 使用 Agg 后端
os.environ.setdefault("MPLBACKEND", "Agg")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import nbformat  # noqa: E402
from nbclient import NotebookClient  # noqa: E402

from gptoy.config import GPConfig  # noqa: E402
from gptoy.evolution import run_gp  # noqa: E402
from gptoy.initialization import initial_population  # noqa: E402
from gptoy.operators import evaluate  # noqa: E402
from gptoy.representation import str_to_tree  # noqa: E402
from gptoy.utils import make_rng  # noqa: E402

NOTEBOOK_PATH = ROOT / "notebooks" / "koza1.ipynb"
LAB_NOTEBOOK_PATH = ROOT / "notebooks" / "GP_learning_lab.ipynb"
NOTES_PATH = ROOT / "notes" / "GP_learning_notes.md"
RESULTS_FOUND_SEED = ROOT / "results" / "found_seed.json"

BANNED_GP_LIBS = ["deap", "gplearn", "sklearn", "pyevolve"]
# 旧 notebook 代码单元禁止出现的「内嵌实现」模式（3.1：不内嵌该步的实现）
BANNED_CELL_PATTERNS = [
    "def ",
    "class ",
    "import random",
    "from random",
    "np.random",
    "random.Random",
    "NotImplementedError",
    "x**4",
    "x**3",
    "x**2",
]

EXPECTED_TITLES = {
    0: "GP 与符号回归",
    1: "表达式树与程序表示",
    2: "种群初始化",
    3: "适应度评价",
    4: "选择机制",
    5: "遗传操作",
    6: "完整进化过程",
    7: "随机性与搜索预算",
    8: "早熟收敛与表达式膨胀",
    9: "小样本、噪声与泛化",
    10: "搜索空间与数值语义的限制",
    11: "GP 失败机制与改进方法",
}
CHAPTER_RE = re.compile(r"^## Chapter (\d+)：(.+)$", re.M)

_passed = []
_failed = []


def _report(section: str, name: str, ok: bool, detail: str = "") -> bool:
    tag = "PASS" if ok else "FAIL"
    line = f"[{section}] {name}: {tag}"
    if detail:
        line += f"  -- {detail}"
    print(line)
    (_passed if ok else _failed).append(f"[{section}] {name}")
    return ok


def _notebook_seed(notebook) -> int | None:
    """从旧 notebook 的代码单元里读取 ``SEED = N``（不依赖未跟踪的结果文件）。"""
    for cell in notebook.cells:
        if cell.cell_type != "code":
            continue
        match = re.search(r"^\s*SEED = (\d+)", "".join(cell.source), re.M)
        if match:
            return int(match.group(1))
    return None


def check_34() -> bool:
    """3.4 环境规定：python ≥ 3.12 且 conda 环境 GPtoy 存在。"""
    ok_version = sys.version_info >= (3, 12)
    _report("3.4", "python 版本 >= 3.12", ok_version, f"当前 {sys.version.split()[0]}")
    has_gptoy_env = False
    try:
        out = subprocess.run(
            ["conda", "env", "list"], capture_output=True, text=True, timeout=120
        )
        has_gptoy_env = any(
            line.strip() and not line.startswith("#") and line.split()[0] == "GPtoy"
            for line in out.stdout.splitlines()
        )
    except (OSError, subprocess.SubprocessError) as exc:
        print(f"  （无法调用 conda：{exc}）")
    _report("3.4", "conda 环境 GPtoy 存在", has_gptoy_env)
    return ok_version and has_gptoy_env


def check_32() -> bool:
    """3.2 从零实现：无 GP 库、目标公式只在 data.py、递归求值冒烟、节点集合断言。"""
    ok_libs = True
    for path in sorted((ROOT / "gptoy").rglob("*.py")):
        src = path.read_text(encoding="utf-8")
        for lib in BANNED_GP_LIBS:
            if re.search(rf"^\s*(import|from)\s+{lib}", src, re.M):
                ok_libs = False
                print(f"  发现禁用库引用：{lib} 在 {path.name}")
    for notebook_path in (NOTEBOOK_PATH, LAB_NOTEBOOK_PATH):
        if not notebook_path.exists():
            continue
        for cell in nbformat.read(notebook_path, as_version=4).cells:
            if cell.cell_type == "code":
                src = "".join(cell.source)
                for lib in ["deap", "gplearn"]:
                    if lib in src:
                        ok_libs = False
                        print(f"  {notebook_path.name} 中发现禁用库引用：{lib}")
    _report("3.2", "无现成遗传编程库", ok_libs)

    # 目标公式唯一合法位置：data.py
    ok_formula = True
    for path in sorted((ROOT / "gptoy").rglob("*.py")):
        src = path.read_text(encoding="utf-8")
        if "x**4" in src and path.name != "data.py":
            ok_formula = False
            print(f"  目标公式出现在 {path.name}（应只在 data.py）")
    ops_src = (ROOT / "gptoy" / "operators.py").read_text(encoding="utf-8")
    if "koza1_target" in ops_src or "x**4" in ops_src:
        ok_formula = False
        print("  fitness 相关文件出现目标公式引用")
    _report("3.2", "目标公式只存在于 data.py（fitness 只吃 (tree, x, y)）", ok_formula)

    # 递归求值冒烟：evaluate 是树递归，结果正确
    cfg = GPConfig()
    t = str_to_tree("(+ x (* x x))", cfg)
    ok_eval = evaluate(t, 3.0) == 12.0
    _report("3.2", "递归求值冒烟（evaluate((+ x (* x x)), 3) == 12）", ok_eval)

    # 节点集合断言：初始种群只含 {+, -, *, /} 与 {x}
    pop = initial_population(cfg, make_rng(0))
    allowed = set(cfg.function_set) | set(cfg.terminal_set)
    ok_nodes = all(
        node.value in allowed for tree in pop for node in _walk(tree)
    ) and len(pop) == cfg.population_size
    _report("3.2", f"初始种群节点集合 ⊆ 函数集∪终止符集 且规模 = {cfg.population_size}", ok_nodes)

    # 核心测试不再依赖未跟踪的 found_seed.json（允许在文档中提及，但不得读取/据此 skip）
    tests_src = "\n".join(
        p.read_text(encoding="utf-8") for p in sorted((ROOT / "tests").glob("*.py"))
    )
    ok_tests = (
        "FOUND_SEED_PATH" not in tests_src
        and 'pytest.skip("results/found_seed.json' not in tests_src
        and "results/found_seed.json 不存在" not in tests_src
    )
    _report("test", "核心测试不依赖 results/found_seed.json", ok_tests)
    return ok_libs and ok_formula and ok_eval and ok_nodes and ok_tests


def check_33(seed: int | None) -> tuple[bool, float | None]:
    """3.3 预算内找回：以 notebook 固定的种子运行 run_gp 并核对指标。"""
    if seed is None:
        _report("3.3", "旧 notebook 中存在固定种子 SEED = N", False)
        return False, None
    cfg = GPConfig(seed=seed)
    result = run_gp(cfg)
    ok_budget = result.n_generations <= cfg.max_generations
    ok_train = result.train_mse < cfg.mse_target
    ok_test = result.test_mse < cfg.mse_target
    _report("3.3", "代数 ≤ 50", ok_budget, f"实际 {result.n_generations}")
    _report("3.3", f"训练 MSE < 1e-6（seed={seed}）", ok_train, f"{result.train_mse:.3e}")
    _report("3.3", f"泛化 MSE < 1e-6（[1,2] 10 点）", ok_test, f"{result.test_mse:.3e}")
    return (ok_budget and ok_train and ok_test), result.gen0_mse


def check_31(seed: int | None, gen0_mse: float | None) -> bool:
    """3.1 教学走得通：结构审查 + nbclient 全量执行 + 时长 + gen-0 未解出。"""
    nb = nbformat.read(NOTEBOOK_PATH, as_version=4)

    headings = [
        c
        for c in nb.cells
        if c.cell_type == "markdown" and re.search(r"^#{1,3} ", "".join(c.source), re.M)
    ]
    ok_md = len(headings) >= 8
    _report("3.1", "markdown 讲解节数 >= 8", ok_md, f"实际 {len(headings)}")

    violations: list[str] = []
    for i, cell in enumerate(nb.cells):
        if cell.cell_type != "code":
            continue
        src = "".join(cell.source)
        for pat in BANNED_CELL_PATTERNS:
            if pat in src:
                violations.append(f"cell#{i} 出现 {pat!r}")
    ok_code = not violations
    _report("3.1", "代码单元无内嵌实现（只调用抽象算子）", ok_code, "; ".join(violations))

    ok_seed = seed is not None and _notebook_seed(nb) == seed
    _report("3.1", f"旧 notebook 固定种子（SEED = {seed}），不依赖 found_seed.json",
            ok_seed,
            "results/found_seed.json 存在与否不影响本验证" if not RESULTS_FOUND_SEED.exists() else "")

    # 全量执行：nbclient 从头到尾跑一遍
    t0 = time.perf_counter()
    try:
        client = NotebookClient(
            nb, timeout=600, kernel_name="gptoy",
            resources={"metadata": {"path": str(ROOT)}},
        )
        client.execute()
        elapsed = time.perf_counter() - t0
        ok_exec = elapsed < 600
        _report("3.1", f"nbclient 全量执行无异常（{elapsed:.1f}s < 10min）", ok_exec)
    except Exception as exc:  # noqa: BLE001
        elapsed = time.perf_counter() - t0
        ok_exec = False
        _report("3.1", "nbclient 全量执行无异常", False, f"{type(exc).__name__}: {exc}")

    ok_gen0 = gen0_mse is not None and gen0_mse > GPConfig().gen0_unsolved_mse
    _report("3.1", "第 0 代未解出（gen0 MSE > 1e-3）", ok_gen0,
            "None" if gen0_mse is None else f"gen0={gen0_mse:.4f}")
    return ok_md and ok_code and ok_seed and ok_exec and ok_gen0


def check_lab() -> bool:
    """学习实验室交付物：Notebook / 笔记 / 章节 / 实验 / 旧案例修复。"""
    ok_exists = LAB_NOTEBOOK_PATH.is_file() and NOTES_PATH.is_file()
    _report("lab", "新 Notebook 与笔记骨架存在", ok_exists)

    ok_chapters = False
    ok_experiments = False
    ok_old_fix = False
    if LAB_NOTEBOOK_PATH.is_file():
        lab = nbformat.read(LAB_NOTEBOOK_PATH, as_version=4)
        md_text = "\n".join(
            "".join(c.source) for c in lab.cells if c.cell_type == "markdown"
        )
        code_text = "\n".join(
            "".join(c.source) for c in lab.cells if c.cell_type == "code"
        )
        titles = {int(m.group(1)): m.group(2).strip() for m in CHAPTER_RE.finditer(md_text)}
        ok_chapters = titles == EXPECTED_TITLES
        _report("lab", "Chapter 0–11 标题完整且正确", ok_chapters,
                f"实际 {len(titles)} 章")
        ok_experiments = all(
            f"### E{i:02d}" in md_text and f"# E{i:02d}" in code_text
            for i in range(1, 11)
        )
        _report("lab", "E01–E10 均有讲解与真实运行入口", ok_experiments)
        ok_kernel = lab.metadata.get("kernelspec", {}).get("name") == "gptoy"
        _report("lab", "新 Notebook 使用 gptoy kernel", ok_kernel)

    if NOTES_PATH.is_file():
        notes = NOTES_PATH.read_text(encoding="utf-8")
        note_titles = {int(m.group(1)): m.group(2).strip() for m in CHAPTER_RE.finditer(notes)}
        ok_notes = note_titles == EXPECTED_TITLES and notes.count("> 待填写") >= 48
        _report("lab", "笔记 12 章齐全且个人填写区留空", ok_notes)
    else:
        ok_notes = False

    old = nbformat.read(NOTEBOOK_PATH, as_version=4)
    cell = next(
        c for c in old.cells
        if c.cell_type == "code" and "apply_variation 输出" in "".join(c.source)
    )
    source = "".join(cell.source)
    ok_old_fix = "20 个父本" in source and "99 个父本" not in source
    _report("lab", "旧 notebook 6.3 父本数量注释已修正（20 个）", ok_old_fix)

    return ok_exists and ok_chapters and ok_experiments and ok_notes and ok_old_fix


def _walk(node):
    yield node
    for child in node.children:
        yield from _walk(child)


def main() -> int:
    print("=" * 64)
    print("GPtoy 项目验证：Success Criteria 3.1–3.4 + 学习实验室交付物")
    print("=" * 64)

    seed = _notebook_seed(nbformat.read(NOTEBOOK_PATH, as_version=4))
    ok33, gen0_mse = check_33(seed)
    ok31 = check_31(seed, gen0_mse)
    ok32 = check_32()
    ok34 = check_34()
    ok_lab = check_lab()

    print("-" * 64)
    all_ok = ok31 and ok32 and ok33 and ok34 and ok_lab
    print(
        f"汇总：3.1={'PASS' if ok31 else 'FAIL'}  3.2={'PASS' if ok32 else 'FAIL'}  "
        f"3.3={'PASS' if ok33 else 'FAIL'}  3.4={'PASS' if ok34 else 'FAIL'}  "
        f"lab={'PASS' if ok_lab else 'FAIL'}"
    )
    if _failed:
        print("未通过项：")
        for item in _failed:
            print(f"  - {item}")
    print("结论：", "全部 PASS" if all_ok else "存在 FAIL，请检查上表")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
