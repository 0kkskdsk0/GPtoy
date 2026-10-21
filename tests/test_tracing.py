"""``gptoy.tracing`` 测试：快照字段、事件真实性、tracer 开关不变性、评估计数。"""

from collections import Counter
from dataclasses import fields, replace

import pytest

from gptoy.config import GPConfig
from gptoy.evolution import EvolutionEngine, run_gp
from gptoy.representation import str_to_tree
from gptoy.tracing import EvolutionTracer, GenerationSnapshot, GeneticEvent, NullTracer

TRACE_CFG = GPConfig(
    seed=2,
    population_size=16,
    max_generations=3,
    max_tree_depth=5,
    min_init_depth=2,
    max_init_depth=4,
    p_crossover=0.6,
    p_mutation=0.3,
)


def test_required_dataclass_fields_exist():
    snapshot_fields = {f.name for f in fields(GenerationSnapshot)}
    assert {
        "generation",
        "best_expression",
        "best_train_mse",
        "best_test_mse",
        "best_fitness",
        "mean_fitness",
        "median_fitness",
        "worst_fitness",
        "mean_size",
        "max_size",
        "mean_depth",
        "max_depth",
        "unique_structures",
        "unique_ratio",
        "evaluations",
        "size_histogram",
        "depth_histogram",
    } <= snapshot_fields

    event_fields = {f.name for f in fields(GeneticEvent)}
    assert {
        "generation",
        "event_type",
        "parent_indices",
        "parent_expressions",
        "child_expressions",
        "crossover_points",
        "mutation_point",
        "parent_fitnesses",
        "child_fitnesses",
        "changed",
        "fallback",
        "elite",
    } <= event_fields


def test_snapshots_match_result():
    tracer = EvolutionTracer()
    result = run_gp(TRACE_CFG, tracer=tracer)
    assert len(tracer.snapshots) == len(result.snapshots)
    assert [s.generation for s in tracer.snapshots] == [
        s.generation for s in result.snapshots
    ]
    assert tracer.latest_snapshot is tracer.snapshots[-1]
    # 累计评估次数：每代 +population_size
    assert [s.evaluations for s in tracer.snapshots] == [
        TRACE_CFG.population_size * (i + 1) for i in range(len(tracer.snapshots))
    ]


def test_snapshot_values_correspond_to_population():
    engine = EvolutionEngine(TRACE_CFG, EvolutionTracer())
    snapshot = engine.initialize()
    best_idx = min(range(len(engine.fitnesses)), key=lambda i: (engine.fitnesses[i], i))
    assert snapshot.best_expression == engine.population[best_idx].sexpr_str()
    assert snapshot.best_fitness == pytest.approx(engine.fitnesses[best_idx])
    assert snapshot.best_train_mse == pytest.approx(engine.fitnesses[best_idx])
    assert snapshot.generation == 0
    assert snapshot.evaluations == TRACE_CFG.population_size


def test_events_are_authentic():
    """父本、操作、交叉/变异点、子代与前后适应度都必须对应真实种群。"""
    tracer = EvolutionTracer()
    engine = EvolutionEngine(TRACE_CFG, tracer)
    engine.initialize()

    prev_population = [tree.sexpr_str() for tree in engine.population]
    prev_fitnesses = list(engine.fitnesses)
    engine.step()
    events = tracer.events_of_generation(1)
    new_population = [tree.sexpr_str() for tree in engine.population]
    new_counter = Counter(new_population)
    next_fitnesses = list(engine.fitnesses)

    assert events, "第 1 代应产生遗传事件"
    # 事件覆盖整代：精英 + 全部子代 = 种群规模
    assert sum(len(e.child_expressions) for e in events) == TRACE_CFG.population_size

    for event in events:
        assert event.generation == 1
        assert len(event.parent_indices) == len(event.parent_expressions)
        assert len(event.child_expressions) == len(event.child_fitnesses)
        for idx, expr, fit in zip(
            event.parent_indices, event.parent_expressions, event.parent_fitnesses
        ):
            assert 0 <= idx < len(prev_population)
            assert expr == prev_population[idx]
            assert fit == pytest.approx(prev_fitnesses[idx])
        # 子代确实出现在新一代种群中
        for child_expr in event.child_expressions:
            assert new_counter[child_expr] > 0
        # 子代适应度与新一代真实适应度一致
        for child_expr, child_fit in zip(event.child_expressions, event.child_fitnesses):
            positions = [i for i, e in enumerate(new_population) if e == child_expr]
            assert any(next_fitnesses[i] == pytest.approx(child_fit) for i in positions)

        if event.event_type == "crossover":
            assert event.crossover_points is not None and len(event.crossover_points) == 2
            assert event.mutation_point is None
            parent1 = str_to_tree(event.parent_expressions[0], TRACE_CFG)
            parent2 = str_to_tree(event.parent_expressions[1], TRACE_CFG)
            assert event.crossover_points[0] in [p for _, p in parent1.iter_nodes()]
            assert event.crossover_points[1] in [p for _, p in parent2.iter_nodes()]
            expected_changed = (
                event.child_expressions[0] != event.parent_expressions[0]
                or event.child_expressions[1] != event.parent_expressions[1]
            )
            assert event.changed is expected_changed
        elif event.event_type == "mutation":
            assert event.mutation_point is not None
            assert event.crossover_points is None
            assert len(event.parent_expressions) == 1
            parent = str_to_tree(event.parent_expressions[0], TRACE_CFG)
            assert event.mutation_point in [p for _, p in parent.iter_nodes()]
            assert event.changed is (event.child_expressions[0] != event.parent_expressions[0])
        elif event.event_type == "reproduction":
            assert event.changed is False
            assert event.child_expressions[0] == event.parent_expressions[0]
        elif event.event_type == "elitism":
            assert event.elite is True and event.changed is False
            assert event.child_expressions[0] == event.parent_expressions[0]
        else:  # pragma: no cover - 事件类型只可能是上面四种
            pytest.fail(f"未知事件类型 {event.event_type}")


def test_elitism_event_matches_previous_best():
    tracer = EvolutionTracer()
    engine = EvolutionEngine(TRACE_CFG, tracer)
    engine.initialize()
    best_idx = min(range(len(engine.fitnesses)), key=lambda i: (engine.fitnesses[i], i))
    best_expression = engine.population[best_idx].sexpr_str()
    best_fitness = engine.fitnesses[best_idx]
    engine.step()

    elitism = tracer.events_of_type("elitism")
    assert len(elitism) == TRACE_CFG.elitism_size
    event = elitism[0]
    assert event.parent_indices == (best_idx,)
    assert event.parent_expressions == (best_expression,)
    assert event.parent_fitnesses == (best_fitness,)
    assert event.child_expressions == (best_expression,)
    assert event.child_fitnesses == (engine.fitnesses[0],)
    assert event.elite is True


def test_all_event_types_observable():
    tracer = EvolutionTracer()
    run_gp(TRACE_CFG, tracer=tracer)
    counts = tracer.counts_by_type()
    assert set(counts) == {"crossover", "mutation", "reproduction", "elitism"}
    assert all(v > 0 for v in counts.values())
    assert len(tracer.events_of_type("crossover")) == counts["crossover"]


def test_fallback_event_recorded():
    """深度上限 1 的极端配置下，交叉必然回退，事件必须如实标记 fallback。"""
    cfg = replace(
        TRACE_CFG,
        max_tree_depth=1,
        min_init_depth=5,
        max_init_depth=5,
        max_generations=1,
        p_crossover=1.0,
        p_mutation=0.0,
    )
    tracer = EvolutionTracer()
    engine = EvolutionEngine(cfg, tracer)
    engine.initialize()
    engine.step()
    crossovers = tracer.events_of_type("crossover")
    assert crossovers
    assert all(e.fallback is True for e in crossovers)
    assert all(e.changed is False for e in crossovers)


def test_tracer_on_off_same_trajectory():
    plain = run_gp(TRACE_CFG)
    traced = run_gp(TRACE_CFG, tracer=EvolutionTracer())
    assert plain.best_mse_history == traced.best_mse_history
    assert plain.best_of_run.sexpr_str() == traced.best_of_run.sexpr_str()


def test_null_tracer_records_nothing():
    null = NullTracer()
    result = run_gp(TRACE_CFG, tracer=null)
    assert null.events == [] and null.snapshots == []
    assert result.snapshots  # 引擎自身的快照仍然存在


def test_snapshot_only_tracer():
    """只保留聚合 snapshots 时不应构造/保存事件，但轨迹与普通运行一致。"""
    tracer = EvolutionTracer(keep_events=False)
    result = run_gp(TRACE_CFG, tracer=tracer)
    plain = run_gp(TRACE_CFG)
    assert tracer.events == []
    assert len(tracer.snapshots) == len(result.snapshots)
    assert result.best_mse_history == plain.best_mse_history
    assert result.best_of_run.sexpr_str() == plain.best_of_run.sexpr_str()


def test_max_events_limit():
    tracer = EvolutionTracer(max_events=5)
    run_gp(TRACE_CFG, tracer=tracer)
    assert len(tracer.events) == 5
    assert tracer.dropped_events > 0


def test_reset():
    tracer = EvolutionTracer()
    run_gp(TRACE_CFG, tracer=tracer)
    assert tracer.snapshots and tracer.events
    tracer.reset()
    assert tracer.snapshots == [] and tracer.events == []
