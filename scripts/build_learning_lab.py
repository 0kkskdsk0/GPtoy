"""组装 GPtoy 学习实验室的 Notebook 与笔记骨架。

用法（在仓库根目录执行）：

    python scripts/build_learning_lab.py            # 目标文件已存在时拒绝覆盖
    python scripts/build_learning_lab.py --force    # 明确覆盖（会丢失已填写的个人笔记！）
    python scripts/build_learning_lab.py --check    # 只检查现有文件与内容定义是否一致

设计约束
--------
- 生成器**默认不覆盖**已存在的 Notebook/笔记文件，避免未经确认地毁掉学习者填写的内容；
- Notebook 的章节标题、核心问题与 notes/GP_learning_notes.md 完全一致（同一份元数据生成）；
- Notebook 只负责教学编排，所有算法都在 gptoy/ 包中实现。
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
# 内容模块与脚本同目录：直接运行脚本时 sys.path[0] 已是 scripts/，这里显式兼容 -m 调用
SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

NOTEBOOK_PATH = ROOT / "notebooks" / "GP_learning_lab.ipynb"
NOTES_PATH = ROOT / "notes" / "GP_learning_notes.md"

from lab_content import CHAPTERS, PARTS, front_matter  # noqa: E402
from lab_content import chapter_0, chapter_1, chapter_2, chapter_3  # noqa: E402
from lab_content_b import CHAPTERS_B  # noqa: E402
from lab_content_b import chapter_4, chapter_5, chapter_6, chapter_7  # noqa: E402
from lab_content_c import CHAPTERS_C  # noqa: E402
from lab_content_c import appendix, chapter_10, chapter_11, chapter_8, chapter_9  # noqa: E402

ALL_CHAPTERS = CHAPTERS + CHAPTERS_B + CHAPTERS_C
CHAPTER_BUILDERS = {
    0: chapter_0,
    1: chapter_1,
    2: chapter_2,
    3: chapter_3,
    4: chapter_4,
    5: chapter_5,
    6: chapter_6,
    7: chapter_7,
    8: chapter_8,
    9: chapter_9,
    10: chapter_10,
    11: chapter_11,
}
EXPECTED_EXPERIMENTS = [f"E{i:02d}" for i in range(1, 11)]


def _validate_metadata() -> None:
    """构建前检查章节元数据完整、编号连续、实验标签齐全。"""
    assert [c["num"] for c in ALL_CHAPTERS] == list(range(12)), "章节编号必须是 0..11"
    for chapter in ALL_CHAPTERS:
        assert chapter["title"], f"Chapter {chapter['num']} 缺少标题"
        assert chapter["core_question"], f"Chapter {chapter['num']} 缺少核心问题"
        assert chapter["source"], f"Chapter {chapter['num']} 缺少源码入口"
    covered = [e for chapter in ALL_CHAPTERS for e in chapter["experiments"]]
    missing = [e for e in EXPECTED_EXPERIMENTS if e not in covered]
    assert not missing, f"以下实验未标注章节：{missing}"
    assert ALL_CHAPTERS[11]["experiments"] == EXPECTED_EXPERIMENTS, "Chapter 11 应综合 E01–E10"


def build_cells() -> list[tuple[str, str]]:
    """按顺序生成全部 Notebook 单元：``[(cell_type, source), ...]``。"""
    _validate_metadata()
    cells: list[tuple[str, str]] = []
    cells.extend(front_matter())

    current_part = None
    for chapter in ALL_CHAPTERS:
        part_index = chapter["part"]
        if part_index != current_part:
            current_part = part_index
            part_no, part_title, part_desc = PARTS[part_index]
            cells.append(
                (
                    "md",
                    f"# {part_no}：{part_title}\n\n"
                    f"*{part_desc}*\n\n"
                    f"> 本部分包含："
                    + "、".join(
                        f"Chapter {c['num']} {c['title']}"
                        for c in ALL_CHAPTERS
                        if c["part"] == part_index
                    ),
                )
            )
        num = chapter["num"]
        chapter_cells = CHAPTER_BUILDERS[num]()
        # 一致性检查：章节标题与元数据完全相同
        expected_heading = f"## Chapter {num}：{chapter['title']}"
        assert any(
            cell_type == "md" and expected_heading in source for cell_type, source in chapter_cells
        ), f"Chapter {num} 的标题与元数据不一致，应包含：{expected_heading}"
        cells.extend(chapter_cells)

    cells.extend(appendix())
    return cells


def build_notebook():
    """构造 nbformat Notebook（不含执行输出与执行计数）。"""
    import nbformat
    from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

    nb_cells = []
    for cell_type, source in build_cells():
        if cell_type == "md":
            nb_cells.append(new_markdown_cell(source))
        else:
            nb_cells.append(new_code_cell(source))
    return new_notebook(
        cells=nb_cells,
        metadata={
            "kernelspec": {
                "display_name": "GPtoy (Python 3.12)",
                "language": "python",
                "name": "gptoy",
            },
            "language_info": {"name": "python", "version": "3.12"},
        },
    )


def _experiment_line(chapter: dict) -> str:
    experiments = chapter["experiments"]
    if not experiments:
        return "无（本章运行 Koza-1 基线）"
    if chapter["num"] == 11:
        return "综合 " + "、".join(experiments)
    return "、".join(experiments)


def build_notes() -> str:
    """生成与 Notebook 章节/核心问题完全一致的笔记骨架（个人区域留空）。"""
    _validate_metadata()
    lines: list[str] = []
    lines.append("# GPtoy 学习笔记（骨架）\n")
    lines.append(
        "> 本文件与 `notebooks/GP_learning_lab.ipynb` 使用同一份章节元数据生成："
        "12 章的编号、标题与「核心问题」完全一致，可独立审阅，也可作为随手笔记的模板。\n"
        ">\n"
        "> **使用方式**：在每章下面第 3–6 节填写自己的预测、配置、观察、解释与总结；"
        "第 1–2 节提供概念与源码索引，供复习时快速定位。\n"
        ">\n"
        "> **个人填写区**：所有 `待填写` 都是留给你的；生成器默认不会覆盖本文件，"
        "只有显式 `--force` 才会重写。\n"
    )

    lines.append("## 总览\n")
    lines.append("| 章节 | 核心问题 | 对应实验 | 源码入口 |")
    lines.append("|---|---|---|---|")
    for chapter in ALL_CHAPTERS:
        lines.append(
            f"| Chapter {chapter['num']}：{chapter['title']} | {chapter['core_question']} "
            f"| {_experiment_line(chapter)} | {chapter['source']} |"
        )
    lines.append("")

    lines.append("## 种子与配置记录区\n")
    lines.append(
        "> 每次实验请记录：日期、章节/实验、配置（种群、代数、深度、概率、阈值）、GP seed、"
        "数据 seed/噪声、实际观察到的关键数字。**先记录种子再看结果**，避免事后挑选。\n"
    )
    lines.append(
        "| 日期 | 章节/实验 | 配置摘要 | GP seed | 数据 seed / 噪声 | 关键观察 |\n"
        "|---|---|---|---|---|---|\n"
        "| 待填写 | 待填写 | 待填写 | 待填写 | 待填写 | 待填写 |\n"
        "| 待填写 | 待填写 | 待填写 | 待填写 | 待填写 | 待填写 |\n"
    )

    lines.append("## 证据分级说明\n")
    lines.append(
        "写笔记时给每句话标注证据级别，避免把猜测写成结论：\n\n"
        "- **观察**：本次运行实际打印/画出的数字与图形（需要种子与配置可追溯）；\n"
        "- **机制解释**：由对照实验支持的因果解释（说明控制了哪些变量、还剩下哪些混淆）；\n"
        "- **猜测**：尚未检验的推测，必须显式标注“猜测”；\n"
        "- **数学事实 / 文献**：外部知识，需注明依据（如“sin 不是有理函数”）。\n\n"
        "特别提醒：训练/测试 MSE 小只能支持“数值拟合”；要声称“结构恢复”，需要额外的"
        "结构判别或数学论证。\n"
    )

    current_part = None
    for chapter in ALL_CHAPTERS:
        if chapter["part"] != current_part:
            current_part = chapter["part"]
            part_no, part_title, _ = PARTS[current_part]
            lines.append(f"\n# {part_no}：{part_title}\n")
        lines.append(f"\n## Chapter {chapter['num']}：{chapter['title']}\n")
        lines.append(f"**核心问题**：{chapter['core_question']}\n")
        lines.append(f"**对应实验**：{_experiment_line(chapter)}\n")

        lines.append("### 1. 核心概念\n")
        for concept in chapter["concepts"]:
            lines.append(f"- {concept}")
        lines.append("")

        lines.append("### 2. 实现机制和源码位置\n")
        lines.append(f"- 主要源码入口：{chapter['source']}")
        lines.append(
            "- 阅读提示：先看函数签名与 docstring，再对照 Notebook 中对应实验的输出，"
            "最后回到实现确认“数字是怎么算出来的”。"
        )
        lines.append("")

        lines.append("### 3. 我的实验问题与预测\n")
        lines.append(
            "> 待填写。建议写：我想验证的问题 → 我的预测与理由 → 我会改哪个变量 → "
            "预期看到什么差异（先写预测，再看结果）。"
        )
        lines.append("")

        lines.append("### 4. 实验配置及观察结果\n")
        lines.append(
            "> 待填写。建议写：种群/代数/深度/概率/阈值、GP seed、数据 seed 与噪声、"
            "关键数字（MSE、首达代、评估次数、unique_ratio、size 等）与图表位置。"
        )
        lines.append("")

        lines.append("### 5. 我的理解、替代解释与证据边界\n")
        lines.append(
            "> 待填写。建议写：支持该解释的证据 → 反对证据/异常点 → 至少一个替代解释 → "
            "目前证据不能区分什么 → 下一步如何区分。"
        )
        lines.append("")

        lines.append("### 6. 本章总结及后续问题\n")
        lines.append("> 待填写。建议写：3 条以内要点 + 2 个可调研的后续问题（含可检验预测）。")
        lines.append("")

    lines.append("\n## 附录：本次学习的环境与复现信息\n")
    lines.append("> 待填写：Python/numpy/matplotlib 版本、kernel 名称、运行日期、总耗时、"
                 "遇到的异常与处理方式。\n")
    return "\n".join(lines)


def _guard(path: Path, force: bool, label: str) -> bool:
    if path.exists() and not force:
        print(
            f"[跳过] {label} 已存在：{path}\n"
            "        为避免覆盖已填写的个人笔记，默认不写入；确认覆盖请加 --force。",
            file=sys.stderr,
        )
        return False
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="覆盖已存在的 Notebook/笔记")
    parser.add_argument("--notebook-only", action="store_true", help="只生成 Notebook")
    parser.add_argument("--notes-only", action="store_true", help="只生成笔记骨架")
    parser.add_argument(
        "--check",
        action="store_true",
        help="只检查元数据、现有文件与生成内容是否一致（忽略执行输出），不写入",
    )
    args = parser.parse_args()

    _validate_metadata()
    cells = build_cells()
    code_cells = [source for kind, source in cells if kind == "code"]
    md_cells = [source for kind, source in cells if kind == "md"]
    print(f"内容检查：{len(cells)} 个单元（markdown {len(md_cells)}，code {len(code_cells)}）")
    for label in EXPECTED_EXPERIMENTS:
        hits = sum(1 for source in code_cells if f"# {label}" in source)
        assert hits >= 1, f"{label} 缺少真实运行入口（代码单元注释 `# {label}`）"
        print(f"  {label}: {hits} 个代码单元")
    for chapter in ALL_CHAPTERS:
        heading = f"## Chapter {chapter['num']}：{chapter['title']}"
        assert any(heading in source for source in md_cells), heading
    print("  Chapter 0–11 标题检查通过")

    if args.check:
        import nbformat

        ok = True
        if NOTEBOOK_PATH.is_file():
            existing = nbformat.read(NOTEBOOK_PATH, as_version=4)
            # 只比较单元类型与 source（忽略 outputs / execution_count），因此执行过的 Notebook
            # 仍然可以检查“源内容是否与生成器一致”。
            existing_cells = [
                ("md" if cell.cell_type == "markdown" else "code", "".join(cell.source))
                for cell in existing.cells
            ]
            notebook_ok = existing_cells == cells
            print(f"Notebook 源单元与内容定义一致（忽略输出）：{notebook_ok}")
            if not notebook_ok:
                mismatch = next(
                    (
                        index
                        for index, (actual, expected) in enumerate(zip(existing_cells, cells))
                        if actual != expected
                    ),
                    None,
                )
                print(
                    f"  首个不一致位置：{mismatch}；"
                    f"期望 {len(cells)} 个单元，实际 {len(existing_cells)} 个"
                )
                ok = False
        else:
            print(f"Notebook 不存在：{NOTEBOOK_PATH}")
            ok = False

        if NOTES_PATH.is_file():
            notes_ok = NOTES_PATH.read_text(encoding="utf-8") == build_notes()
            print(f"笔记内容与生成器一致：{notes_ok}")
            ok = ok and notes_ok
        else:
            print(f"笔记不存在：{NOTES_PATH}")
            ok = False
        return 0 if ok else 1

    written = []
    if not args.notes_only:
        if _guard(NOTEBOOK_PATH, args.force, "Notebook"):
            import nbformat

            NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
            nbformat.write(build_notebook(), NOTEBOOK_PATH)
            written.append(str(NOTEBOOK_PATH))
    if not args.notebook_only:
        if _guard(NOTES_PATH, args.force, "笔记骨架"):
            NOTES_PATH.parent.mkdir(parents=True, exist_ok=True)
            NOTES_PATH.write_text(build_notes(), encoding="utf-8")
            written.append(str(NOTES_PATH))

    for item in written:
        print(f"已写入：{item}")
    if not written:
        print("未写入任何文件。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
