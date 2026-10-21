"""教学交付物静态检查（快速）：Notebook 结构、实验入口、笔记一致性、旧案例修复。"""

import re
import subprocess
import sys
from pathlib import Path

import nbformat
import pytest

ROOT = Path(__file__).resolve().parents[1]
LAB_PATH = ROOT / "notebooks" / "GP_learning_lab.ipynb"
OLD_PATH = ROOT / "notebooks" / "koza1.ipynb"
NOTES_PATH = ROOT / "notes" / "GP_learning_notes.md"
README_PATH = ROOT / "README.md"

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
EXPECTED_EXPERIMENTS = [f"E{i:02d}" for i in range(1, 11)]
CHAPTER_RE = re.compile(r"^## Chapter (\d+)：(.+)$", re.M)
# 核心问题可能被 Markdown 软换行拆成多行：匹配到空行为止，再归一化空白
CORE_RE = re.compile(r"^\*\*核心问题\*\*：(.+?)\n\s*\n", re.M | re.S)


def _core_questions(text: str) -> set[str]:
    return {re.sub(r"\s+", "", match.group(1)) for match in CORE_RE.finditer(text)}


@pytest.fixture(scope="module")
def notebook():
    return nbformat.read(LAB_PATH, as_version=4)


def _markdown_texts(nb) -> list[tuple[int, str]]:
    return [(i, "".join(c.source)) for i, c in enumerate(nb.cells) if c.cell_type == "markdown"]


def _chapter_blocks(nb) -> dict[int, str]:
    """把 Notebook 按章节标题切成 ``{章节号: 该章全部 markdown 文本}``。"""
    md_cells = _markdown_texts(nb)
    positions = []
    for cell_index, text in md_cells:
        for match in CHAPTER_RE.finditer(text):
            positions.append((int(match.group(1)), cell_index))
    positions.sort()
    blocks: dict[int, str] = {}
    for order, (num, start) in enumerate(positions):
        end = positions[order + 1][1] if order + 1 < len(positions) else len(nb.cells)
        blocks[num] = "\n".join(
            "".join(cell.source) for cell in nb.cells[start:end] if cell.cell_type == "markdown"
        )
    return blocks


def test_deliverables_exist():
    for path in (LAB_PATH, OLD_PATH, NOTES_PATH, README_PATH):
        assert path.is_file(), f"缺少交付物：{path}"


def test_notebook_is_valid_and_uses_gptoy_kernel(notebook):
    nbformat.validate(notebook)
    kernelspec = notebook.metadata.get("kernelspec", {})
    assert kernelspec.get("name") == "gptoy"
    assert "GPtoy" in kernelspec.get("display_name", "")


def test_three_parts_present(notebook):
    text = "\n".join(text for _, text in _markdown_texts(notebook))
    assert "# 第一篇：GP 的经典运行机制" in text
    assert "# 第二篇：GP 的典型失败" in text
    assert "# 第三篇：从实验走向研究" in text


def test_all_chapters_in_order(notebook):
    md_cells = _markdown_texts(notebook)
    found: dict[int, str] = {}
    order: list[int] = []
    for _, text in md_cells:
        for match in CHAPTER_RE.finditer(text):
            num, title = int(match.group(1)), match.group(2).strip()
            found[num] = title
            order.append(num)
    assert found == EXPECTED_TITLES
    # 标题在文档中按 0..11 出现（不重复）
    assert order == list(range(12))


def test_every_chapter_has_required_sections(notebook):
    blocks = _chapter_blocks(notebook)
    assert set(blocks) == set(range(12))
    required_markers = [
        "**核心问题**",
        "### 概念与数学定义",
        "### 具体小例子",
        "### 源码入口",
        "我的预测（请填写）",
        "观察与解释（请填写）",
        "### 本章笔记（请填写）",
    ]
    for num, block in blocks.items():
        for marker in required_markers:
            assert marker in block, f"Chapter {num} 缺少教学要素：{marker}"


def test_part_headings_appear_once(notebook):
    """三篇标题只应由 builder 生成一次，章节内容里不得重复。"""
    md_text = "\n".join(text for _, text in _markdown_texts(notebook))
    for part in ("第一篇", "第二篇", "第三篇"):
        headings = re.findall(rf"^#{{1,2}} {part}", md_text, re.M)
        assert len(headings) == 1, f"{part} 标题出现 {len(headings)} 次（应只有 builder 生成的 1 次）"


def test_code_cells_do_not_prewrite_research_conclusions(notebook):
    """代码单元只提供数据，不得替学习者写“这说明/结论是”式的结论。"""
    forbidden_phrases = (
        "这说明",
        "结论级别",
        "结论是",
        "因此证明",
        "这证明了",
        "由此可见",
    )
    for index, cell in enumerate(notebook.cells):
        if cell.cell_type != "code":
            continue
        source = "".join(cell.source)
        for phrase in forbidden_phrases:
            assert phrase not in source, f"cell#{index} 出现预写结论短语 {phrase!r}"


def test_learning_notes_placeholders(notebook):
    blocks = _chapter_blocks(notebook)
    for num, block in blocks.items():
        for section in (
            "**1. 核心概念",
            "**2. 实现机制与源码位置",
            "**3. 我的实验问题与预测",
            "**4. 实验配置及观察结果",
            "**5. 我的理解、替代解释与证据边界",
            "**6. 本章总结与后续问题",
        ):
            assert block.count(section) == 1, f"Chapter {num} 的笔记区不完整：{section}"
        assert "待填写" in block


def test_experiments_have_markdown_and_code_entries(notebook):
    md_text = "\n".join(text for _, text in _markdown_texts(notebook))
    blocks = _chapter_blocks(notebook)
    for number, label in enumerate(EXPECTED_EXPERIMENTS, start=1):
        assert f"### {label}" in md_text, f"缺少实验讲解：{label}"
        assert label in blocks[number], f"{label} 应位于 Chapter {number} 内"
        code_cells = [
            cell
            for cell in notebook.cells
            if cell.cell_type == "code" and f"# {label}" in "".join(cell.source)
        ]
        assert code_cells, f"{label} 缺少真实运行入口（代码单元注释 `# {label}`）"


def test_experiment_code_located_in_matching_chapter(notebook):
    heading_positions = {}
    for index, cell in enumerate(notebook.cells):
        if cell.cell_type != "markdown":
            continue
        match = CHAPTER_RE.search("".join(cell.source))
        if match:
            heading_positions[int(match.group(1))] = index
    assert set(heading_positions) == set(range(12))

    boundaries = [
        heading_positions[num] for num in range(12)
    ] + [len(notebook.cells)]
    assert boundaries == sorted(boundaries)
    for number, label in enumerate(EXPECTED_EXPERIMENTS, start=1):
        positions = [
            index
            for index, cell in enumerate(notebook.cells)
            if cell.cell_type == "code" and f"# {label}" in "".join(cell.source)
        ]
        assert positions, label
        start = heading_positions[number]
        end = heading_positions.get(number + 1, len(notebook.cells))
        assert start < end
        for position in positions:
            assert start <= position < end, (
                f"{label} 的代码单元不在 Chapter {number} 范围内"
            )


def test_no_prefilled_personal_conclusions(notebook):
    md_text = "\n".join(text for _, text in _markdown_texts(notebook))
    assert md_text.count("我的预测（请填写）") >= 10
    assert md_text.count("观察与解释（请填写）") >= 10
    assert md_text.count("待填写") >= 12


def test_notes_match_notebook_titles_and_questions(notebook):
    notes = NOTES_PATH.read_text(encoding="utf-8")
    nb_text = "\n".join(text for _, text in _markdown_texts(notebook))

    nb_titles = {int(m.group(1)): m.group(2).strip() for m in CHAPTER_RE.finditer(nb_text)}
    note_titles = {int(m.group(1)): m.group(2).strip() for m in CHAPTER_RE.finditer(notes)}
    assert note_titles == nb_titles

    nb_questions = _core_questions(nb_text)
    note_questions = _core_questions(notes)
    assert note_questions == nb_questions
    assert len(note_questions) == 12


def test_notes_skeleton_sections():
    notes = NOTES_PATH.read_text(encoding="utf-8")
    headings = [
        "### 1. 核心概念",
        "### 2. 实现机制和源码位置",
        "### 3. 我的实验问题与预测",
        "### 4. 实验配置及观察结果",
        "### 5. 我的理解、替代解释与证据边界",
        "### 6. 本章总结及后续问题",
    ]
    for heading in headings:
        assert notes.count(heading) == 12, f"笔记章节数不对：{heading}"
    assert notes.count("> 待填写") >= 12 * 4
    # 总览、种子/配置记录、证据分级
    assert "## 总览" in notes
    assert "## 种子与配置记录区" in notes
    assert "## 证据分级说明" in notes
    for label in EXPECTED_EXPERIMENTS:
        assert label in notes


def test_old_notebook_6_3_comment_fixed():
    old = nbformat.read(OLD_PATH, as_version=4)
    cell = next(
        c for c in old.cells if c.cell_type == "code" and "apply_variation 输出" in "".join(c.source)
    )
    source = "".join(cell.source)
    assert "20 个父本" in source
    assert "99 个父本" not in source
    # 原案例结构保留：仍然 28 个单元
    assert len(old.cells) == 28


def test_readme_covers_required_topics():
    readme = README_PATH.read_text(encoding="utf-8")
    for keyword in (
        "GP_learning_lab.ipynb",
        "GP_learning_notes.md",
        "pytest -q -m slow",
        "保存",
        "复现",
        "限制",
        "seed=3",
        "find_seed.py",
        "保护除法",
        "kernel",
    ):
        assert keyword in readme, f"README 缺少主题：{keyword}"


def test_no_third_party_gp_library():
    banned = ("deap", "gplearn", "pyevolve")
    for path in (ROOT / "gptoy").rglob("*.py"):
        source = path.read_text(encoding="utf-8")
        for lib in banned:
            assert not re.search(rf"^\s*(import|from)\s+{lib}", source, re.M), f"{path.name} 引用了 {lib}"
    operators_source = (ROOT / "gptoy" / "operators.py").read_text(encoding="utf-8")
    assert "koza1_target" not in operators_source
    assert "x**4" not in operators_source


def test_generator_check_passes():
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "build_learning_lab.py"), "--check"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=120,
    )
    assert result.returncode == 0, result.stderr + result.stdout
    assert "Chapter 0" in result.stdout
    assert "E10" in result.stdout
