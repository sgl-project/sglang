"""Exercise actual run_batch routing without accelerator/model initialization."""
import ast
import time
from contextlib import nullcontext
from functools import partial
from pathlib import Path
from types import SimpleNamespace as NS

import pytest


@pytest.mark.parametrize("speculative", [False, True])
def test_only_speculative_forward_publishes_lengths(speculative):
    path = Path(__file__).parents[4] / "python/sglang/srt/managers/scheduler.py"
    cls = next(n for n in ast.parse(path.read_text()).body
               if isinstance(n, ast.ClassDef) and n.name == "Scheduler")
    fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "run_batch")
    fn.decorator_list = []
    namespace = {"time": time, "partial": partial, "_is_hip": False,
                 "DisaggregationMode": NS(NULL=0, PREFILL=1)}
    resolved, relayed, published = [], [], []
    namespace["resolve_forward_inputs"] = lambda b, f: resolved.append(b)
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), fn], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)

    class NoDeviceAddition:
        def __add__(self, value):
            raise AssertionError("Non-spec publication must not enqueue an unused device add")

    result = NS(extra_keep_alive_refs=[], delay_sample_func=None,
                next_draft_input=NS(dsa_topk_indices=None), copy_to_cpu=lambda **kw: None)
    indices, verified_lengths = object(), object()
    batch = NS(forward_mode=NS(is_prebuilt=lambda: False),
               spec_algorithm=NS(is_none=lambda: not speculative, supports_grammar_overlap=lambda: False),
               req_pool_indices=indices, seq_lens=NoDeviceAddition(),
               return_logprob=False, return_hidden_states=False, input_ids=object(), reqs=[])
    def forward(batch, **kwargs):
        if speculative:
            kwargs["on_publish"](verified_lengths)
        else:
            assert "on_publish" not in kwargs
        return result
    stream = NS(wait_stream=lambda other: None)
    scheduler = NS(
        forward_ct=0, _sched_idled=False, scripted_scheduler_hook=None,
        profiler_manager=NS(_profile_batch_predicate=lambda b: None), forward_sleep_time=None,
        disaggregation_mode=0, is_generation=True, enable_overlap=True,
        future_map=NS(resolve_seq_lens_cpu=lambda b: None,
                      publish=lambda idx, lengths: published.append((idx, lengths))),
        _confidence_budget_prepare=None, forward_stream_ctx=nullcontext(),
        forward_stream=stream, schedule_stream=stream, copy_stream=stream,
        copy_stream_ctx=nullcontext(), _forward_isolation=lambda *a, **kw: nullcontext(),
        model_worker=NS(forward_batch_generation=forward), enable_unified_memory=False,
        device_module=NS(Event=object),
        _relay_forward_payload=lambda idx, r: relayed.append((idx, r)),
        _maybe_report_active_ranks=lambda: None,
    )
    assert namespace["run_batch"](scheduler, batch) is result
    assert resolved == [batch]
    assert relayed == [(indices, result)]
    assert published == ([(indices, verified_lengths)] if speculative else [])
    assert batch.input_ids is None
