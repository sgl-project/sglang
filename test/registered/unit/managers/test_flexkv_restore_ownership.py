"""Exercise the real scheduling methods without loading the GPU serving stack."""

import ast
from enum import Enum, auto
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def _load_method(filename, class_name, method_name, namespace):
    # Execute the complete production method, not a copy of the guard logic.
    path = Path(__file__).resolve().parents[4] / "python/sglang/srt/managers" / filename
    tree = ast.parse(path.read_text())
    cls = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    method = next(
        node
        for node in cls.body
        if isinstance(node, ast.FunctionDef) and node.name == method_name
    )
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            method,
        ],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return namespace[method_name]


class AddReqResult(Enum):
    CONTINUE = auto()
    OTHER = auto()
    NO_TOKEN = auto()


def _scheduler_case(*, chunked=False, flexkv=False):
    req = SimpleNamespace(
        rid="restore",
        init_next_round_input=MagicMock(),
        mamba_pool_idx=None,
        beam_group=None,
        kv=SimpleNamespace(holds_mamba=False),
    )
    leased = {req.rid}
    cache = SimpleNamespace(
        has_uncommitted_restore=lambda r: r.rid in leased,
        check_hicache_events=MagicMock(),
        check_prefetch_progress=MagicMock(return_value=True),
        pop_prefetch_loaded_span=MagicMock(return_value=(0, None)),
    )
    adder = SimpleNamespace(
        can_run_list=[],
        add_one_req=MagicMock(return_value=AddReqResult.OTHER),
        add_chunked_req=MagicMock(return_value=None),
    )
    scheduler = SimpleNamespace(
        grammar_manager=SimpleNamespace(has_waiting_grammars=lambda: False),
        tree_cache=cache,
        enable_hierarchical_cache=False,
        enable_unified_cache_external_linker=False,
        enable_priority_preemption=False,
        is_hybrid_swa=False,
        waiting_queue=[] if chunked else [req],
        chunked_req=req if chunked else None,
        min_free_slots_delayer=None,
        get_num_allocatable_reqs=lambda *_args, **_kwargs: 8,
        policy=SimpleNamespace(calc_priority=MagicMock()),
        chunked_prefill_size=16,
        dynamic_chunk_sizer=None,
        tp_worker=SimpleNamespace(model_runner=SimpleNamespace(attn_backend=object())),
        page_size=4,
        token_to_kv_pool_allocator=object(),
        new_token_ratio_tracker=SimpleNamespace(current=0.5),
        max_prefill_tokens=32,
        is_mixed_chunk=False,
        priority_scheduling_preemption_threshold=0,
        max_prefill_bs=8,
        max_running_requests=8,
        dllm_config=None,
        enable_lora=False,
        req_to_token_pool=SimpleNamespace(mamba_allocator=None),
        enable_hicache_storage=False,
        enable_flexkv=flexkv,
        disaggregation_mode=None,
        truncation_align_size=None,
    )
    run = _load_method(
        "scheduler.py",
        "Scheduler",
        "_get_new_batch_prefill_raw",
        {
            "PrefillAdder": lambda *_args, **_kwargs: adder,
            "get_memory": lambda: SimpleNamespace(enable_flexkv=flexkv),
            "get_schedule": lambda: SimpleNamespace(prefill_max_requests=None),
            "TEST_RETRACT": False,
            "AddReqResult": AddReqResult,
            "DisaggregationMode": SimpleNamespace(PREFILL="prefill"),
        },
    )
    running = SimpleNamespace(reqs=[], batch_is_full=False, is_empty=lambda: True)
    return req, leased, adder, lambda: run(scheduler, None, running)


def test_idle_flexkv_retries_admission_after_no_token():
    _, leased, adder, run = _scheduler_case(flexkv=True)
    leased.clear()
    adder.add_one_req.side_effect = [AddReqResult.NO_TOKEN, AddReqResult.OTHER]
    run()
    run()
    assert adder.add_one_req.call_count == 2


@pytest.mark.parametrize("chunked", [False, True])
def test_scheduler_defers_restore_rematch_then_resumes_after_commit(chunked):
    req, leased, adder, run = _scheduler_case(chunked=chunked)
    run()
    req.init_next_round_input.assert_not_called()
    adder.add_one_req.assert_not_called()
    adder.add_chunked_req.assert_not_called()
    leased.clear()
    run()
    req.init_next_round_input.assert_called_once()
    if chunked:
        adder.add_chunked_req.assert_called_once_with(req)
    else:
        adder.add_one_req.assert_called_once()


def test_admission_cannot_reject_after_allocating_restore_slots():
    req, leased, adder, run = _scheduler_case()
    leased.clear()

    def load_then_reject(*_args, **_kwargs):
        leased.add(req.rid)
        return AddReqResult.OTHER

    adder.add_one_req.side_effect = load_then_reject
    with pytest.raises(RuntimeError, match="rejected after storage load-back"):
        run()
    assert req.rid in leased


@pytest.mark.parametrize("cache_aware", [False, True])
def test_priority_matching_preserves_uncommitted_restore(cache_aware):
    req = SimpleNamespace(
        rid="restore",
        origin_input_ids=list(range(4)),
        output_ids=[],
        prefix_indices=list(range(4)),
        extra_key=None,
        cache_salt=None,
    )
    leased = {req.rid}
    matcher = MagicMock()
    cache = SimpleNamespace(
        has_uncommitted_restore=lambda r: r.rid in leased,
        supports_fast_match_prefix=lambda: True,
    )
    if cache_aware:
        policy = SimpleNamespace(tree_cache=cache, waiting_queue_radix_tree=MagicMock())
        run = _load_method(
            "schedule_policy.py",
            "SchedulePolicy",
            "_compute_prefix_matches",
            {
                "match_prefix_for_req": matcher,
                "IN_BATCH_PREFIX_CACHING_CHECK_THRESHOLD": 0,
            },
        )
        invoke = lambda: run(policy, [req], None)
    else:

        class CacheAgnosticPolicy(Enum):
            FCFS = auto()

        policy = SimpleNamespace(
            tree_cache=cache,
            policy=CacheAgnosticPolicy.FCFS,
            _determine_active_policy=lambda _q: CacheAgnosticPolicy.FCFS,
            enable_priority_scheduling=False,
        )
        run = _load_method(
            "schedule_policy.py",
            "SchedulePolicy",
            "calc_priority",
            {
                "match_prefix_for_req": matcher,
                "CacheAwarePolicy": type("CacheAwarePolicy", (), {}),
                "CacheAgnosticPolicy": CacheAgnosticPolicy,
                "get_disagg": lambda: SimpleNamespace(disaggregation_mode=None),
            },
        )
        invoke = lambda: run(policy, [req])
    invoke()
    matcher.assert_not_called()
    leased.clear()
    invoke()
    matcher.assert_called_once()
