"""``gptoy.operators`` 与 ``gptoy.data`` 冒烟测试。"""

import numpy as np
import pytest

from gptoy.config import GPConfig
from gptoy.data import koza1_target, make_koza1_data
from gptoy.operators import evaluate, fitness, p_div
from gptoy.representation import str_to_tree

CFG = GPConfig()
X_TRAIN, Y_TRAIN, X_TEST, Y_TEST = make_koza1_data(CFG)

# 目标表达式对应的树：(x^4 + x^3 + x^2) + x，全部由 {+, -, *, /} 与 {x} 组成
TARGET_TREE = str_to_tree(
    "(+ (+ (+ (* (* x x) (* x x)) (* x (* x x))) (* x x)) x)", CFG
)


def test_protected_div():
    # 标量：除数为 0 → 1.0；正常除法正确
    assert p_div(1.0, 0.0) == 1.0
    assert p_div(6.0, 2.0) == 3.0
    # ndarray：逐元素保护，无 NaN/Inf
    out = p_div(np.array([1.0, 6.0, -3.0]), np.array([0.0, 2.0, 0.0]))
    assert np.allclose(out, [1.0, 3.0, 1.0])
    assert not np.any(np.isnan(out)) and not np.any(np.isinf(out))


def test_evaluate_scalar():
    t = str_to_tree("(+ x x)", CFG)
    assert evaluate(t, 3.0) == 6.0
    # 保护除法：x=0 时 x/x = 1.0（不产生 NaN）
    assert evaluate(str_to_tree("(/ x x)", CFG), 0.0) == 1.0


def test_evaluate_vector():
    t = str_to_tree("(* x x)", CFG)
    xs = np.array([0.0, 1.0, 2.0, 3.0])
    out = evaluate(t, xs)
    assert out.shape == xs.shape
    assert np.allclose(out, xs**2)


def test_fitness_true_expression():
    # 目标表达式在训练集上 MSE ≈ 0
    assert fitness(TARGET_TREE, X_TRAIN, Y_TRAIN) < 1e-12


def test_fitness_not_hardcoded():
    # fitness 只依赖传入的 (tree, x, y)：改 y 则 MSE 变
    mse_orig = fitness(TARGET_TREE, X_TRAIN, Y_TRAIN)
    mse_shifted = fitness(TARGET_TREE, X_TRAIN, Y_TRAIN + 1.0)
    assert mse_shifted > mse_orig > 0 or mse_orig < 1e-12
    assert mse_shifted != mse_orig


def test_data_generation():
    assert X_TRAIN.shape == (20,) and X_TEST.shape == (10,)
    assert X_TRAIN[0] == -1.0 and X_TRAIN[-1] == 1.0
    assert X_TEST[0] == 1.0 and X_TEST[-1] == 2.0
    assert np.allclose(Y_TRAIN, koza1_target(X_TRAIN))
    assert np.allclose(Y_TEST, koza1_target(X_TEST))
    assert koza1_target(2.0) == 16 + 8 + 4 + 2  # 标量输入
