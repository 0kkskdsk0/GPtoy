"""离线种子搜索（Success Criteria 3.1 / 3.3 的前置步骤）。

在候选种子范围内运行 ``run_gp``，寻找同时满足以下条件的种子：

1. 第 0 代 best MSE > ``config.gen0_unsolved_mse`` —— 未在第 0 代解出，进化可观测（3.1）；
2. 训练 MSE < ``mse_target`` 且 泛化 MSE < ``mse_target`` —— 预算内找回 + 泛化（3.3）。

两阶段策略：
- 阶段 1（筛选）：用 ``--gens``（默认 30）代快速扫描候选种子（非收敛种子更快失败）；
- 阶段 2（确认）：对通过筛选的种子用完整预算（50 代）复跑确认。

教学偏好：在全部通过的种子中，优先选择「首达代 ∈ [5, 25]」的种子
（既能看到若干代的进化过程，又不会让 notebook 运行太久）。

把最佳种子与运行统计写入 ``results/found_seed.json``，供 notebook 固定使用。

用法（在仓库根目录执行）：
    python scripts/find_seed.py [--max-seed 200] [--gens 30] [--workers 8]
"""

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gptoy.config import GPConfig  # noqa: E402
from gptoy.evolution import RunResult, run_gp  # noqa: E402

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
FOUND_SEED_PATH = RESULTS_DIR / "found_seed.json"


def _run_one(seed: int, gens: int) -> dict:
    """在单个 worker 中运行一次 GP，返回可序列化的统计字典。"""
    config = GPConfig(seed=seed, max_generations=gens)
    result: RunResult = run_gp(config)
    return {
        "seed": seed,
        "gen0_mse": result.gen0_mse,
        "generation_found": result.generation_found,
        "train_mse": result.train_mse,
        "test_mse": result.test_mse,
        "n_generations": result.n_generations,
        "total_seconds": result.total_seconds,
    }


def _passes(stat: dict, config: GPConfig, max_gens: int) -> bool:
    return (
        stat["gen0_mse"] > config.gen0_unsolved_mse
        and stat["generation_found"] is not None
        and stat["generation_found"] <= max_gens
        and stat["train_mse"] < config.mse_target
        and stat["test_mse"] < config.mse_target
    )


def _scan(seeds: list[int], gens: int, workers: int) -> list[dict]:
    stats: list[dict] = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_run_one, s, gens): s for s in seeds}
        for i, future in enumerate(as_completed(futures), start=1):
            seed = futures[future]
            try:
                stat = future.result()
            except Exception as exc:  # 单种子失败不影响整体搜索
                print(f"  seed={seed}: 运行失败（{exc}）", flush=True)
                continue
            stats.append(stat)
            found = stat["generation_found"]
            status = f"found@gen={found}" if found is not None else "not-found"
            print(
                f"  [{i:4d}/{len(seeds)}] seed={seed:4d} gen0={stat['gen0_mse']:.4f} "
                f"{status} train={stat['train_mse']:.3e} test={stat['test_mse']:.3e} "
                f"t={stat['total_seconds']:.1f}s",
                flush=True,
            )
    return stats


def _teaching_rank(stat: dict) -> tuple:
    """教学偏好排序：首达代 ∈ [5, 25] 优先（可见多代进化），再按首达代、种子号。"""
    gen = stat["generation_found"] or 10**9
    prefer_visible = 0 if 5 <= gen <= 25 else 1
    return (prefer_visible, gen, stat["seed"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-seed", type=int, default=200, help="扫描种子范围 [0, max_seed]")
    parser.add_argument("--gens", type=int, default=30, help="筛选阶段的最大代数")
    parser.add_argument("--workers", type=int, default=8, help="并行 worker 数")
    args = parser.parse_args()

    config = GPConfig()
    seeds = list(range(args.max_seed + 1))
    print(f"阶段 1：用 {args.gens} 代筛选种子 0..{args.max_seed}（{len(seeds)} 个）")
    screen_stats = _scan(seeds, args.gens, args.workers)

    candidates = [s for s in screen_stats if _passes(s, config, args.gens)]
    print(f"筛选通过：{len(candidates)} 个种子")

    confirmed: list[dict] = []
    if candidates:
        print(f"阶段 2：对通过筛选的种子用完整预算（{config.max_generations} 代）确认")
        confirmed = _scan([c["seed"] for c in candidates], config.max_generations, args.workers)
        confirmed = [c for c in confirmed if _passes(c, config, config.max_generations)]
    print(f"完整预算确认通过：{len(confirmed)} 个种子")

    if not confirmed:
        print("ERROR: 未找到满足全部条件的种子，请调整参数后重试", file=sys.stderr)
        return 1

    best = min(confirmed, key=_teaching_rank)
    print(f"选定种子：{best['seed']}（教学偏好排序后最优）")
    print(json.dumps(best, ensure_ascii=False, indent=2))

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "seed": best["seed"],
        "config": {
            "population_size": config.population_size,
            "max_generations": config.max_generations,
            "mse_target": config.mse_target,
            "gen0_unsolved_mse": config.gen0_unsolved_mse,
        },
        "run": best,
    }
    FOUND_SEED_PATH.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"已写入 {FOUND_SEED_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
