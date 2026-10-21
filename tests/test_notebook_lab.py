"""重型测试（``-m slow``）：用 nbclient 从头执行教学 Notebook。

- ``GP_learning_lab.ipynb``：Chapter 0–11、E01–E10 全量执行，要求无异常、有真实图表输出；
- ``koza1.ipynb``：原教学案例仍可执行，且 6.3 的父本数量输出为 20。
"""

import os
import time
from pathlib import Path

import nbformat
import pytest
from nbclient import NotebookClient

os.environ.setdefault("MPLBACKEND", "Agg")

ROOT = Path(__file__).resolve().parents[1]
LAB_PATH = ROOT / "notebooks" / "GP_learning_lab.ipynb"
OLD_PATH = ROOT / "notebooks" / "koza1.ipynb"
EXPECTED_EXPERIMENTS = [f"E{i:02d}" for i in range(1, 11)]


def _execute(path: Path, timeout: int):
    notebook = nbformat.read(path, as_version=4)
    client = NotebookClient(
        notebook,
        timeout=timeout,
        kernel_name="gptoy",
        resources={"metadata": {"path": str(ROOT)}},
    )
    started = time.perf_counter()
    client.execute()
    return notebook, time.perf_counter() - started


def _errors(notebook) -> list:
    return [
        output
        for cell in notebook.cells
        if cell.cell_type == "code"
        for output in cell.get("outputs", [])
        if output.get("output_type") == "error"
    ]


@pytest.mark.slow
def test_learning_lab_executes_end_to_end():
    notebook, elapsed = _execute(LAB_PATH, timeout=1200)
    print(f"\nGP_learning_lab.ipynb 执行耗时：{elapsed:.1f}s")

    errors = _errors(notebook)
    assert not errors, f"Notebook 执行出现异常：{errors[:1]}"
    assert elapsed < 900, f"Notebook 执行超过 15 分钟：{elapsed:.1f}s"

    code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
    assert len(code_cells) >= 30

    image_outputs = sum(
        1
        for cell in code_cells
        for output in cell.get("outputs", [])
        if "image/png" in output.get("data", {})
    )
    assert image_outputs >= 10, f"图表输出过少：{image_outputs}"

    for label in EXPECTED_EXPERIMENTS:
        cells = [cell for cell in code_cells if f"# {label}" in "".join(cell.source)]
        assert cells, f"{label} 缺少可运行代码单元"
        assert any(cell.get("outputs") for cell in cells), f"{label} 没有任何真实输出"
        for cell in cells:
            stream = "".join(
                "".join(output.get("text", []))
                for output in cell.get("outputs", [])
                if output.get("output_type") == "stream"
            )
            images = [
                output
                for output in cell.get("outputs", [])
                if "image/png" in output.get("data", {})
            ]
            assert stream.strip() or images, f"{label} 的输出为空"


@pytest.mark.slow
def test_old_koza1_notebook_still_executes():
    notebook, elapsed = _execute(OLD_PATH, timeout=600)
    print(f"\nkoza1.ipynb 执行耗时：{elapsed:.1f}s")
    assert elapsed < 300
    errors = _errors(notebook)
    assert not errors, f"旧 Notebook 执行出现异常：{errors[:1]}"

    cell = next(
        c
        for c in notebook.cells
        if c.cell_type == "code" and "apply_variation 输出" in "".join(c.source)
    )
    text = "".join(
        "".join(output.get("text", []))
        for output in cell.get("outputs", [])
        if output.get("output_type") == "stream"
    )
    assert "20" in text, f"6.3 父本数量输出不正确：{text!r}"
