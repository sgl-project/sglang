"""CPU-only runtime-flow tests for the BF16 trtllm MoE-LoRA two-stream fork.

``moe_overlap.py`` no longer holds a second copy of the dispatch: it only decides
whether to fork, and every stream operation now lives in the single-stream body of
``fused_experts_none_to_experimental_sgl_trtllm_bf16_lora`` behind
``gate_up_lora_stream is not None``. The B200 equivalence test never sets it, so
without this file the join, the main-stream hoists and the fork are uncovered and
a decode-corrupting reorder merges green.

Recording fakes make the schedule observable with no CUDA, the same way
``test_experimental_sgl_marlin_runtime_unit.py`` does for the Marlin runner. Three
things are asserted, all ordering and ownership, never numerics:

- the routing pre-warm and the shrink intermediate are allocated on the MAIN
  stream (allocating inside the side-stream context during cuda-graph capture
  reuses pool blocks with no cross-stream edge -> '!!!!' decode corruption);
- the gate_up shrink/expand runs on the side stream and asks for the half swap;
- the main stream joins the side stream before anything reads the delta -- the
  ``torch.cat`` on the rank > 64 path as well as the MoE op itself.
"""

from __future__ import annotations

import contextlib
import sys
import types
from types import SimpleNamespace

import pytest
import torch

from sglang.srt.lora.trtllm_lora_temp import lora_dispatch
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

# Both of these are monkeypatched per flow; hold the pristine callables so running
# two flows in one test can never chain the recording wrappers.
_ORIGINAL_NEW_EMPTY = torch.Tensor.new_empty
_ORIGINAL_CAT = torch.cat

HIDDEN = 8
INTER = 8
TOP_K = 2
NUM_EXPERTS = 4


def _stub_module(monkeypatch, name: str, **attributes):
    parts = name.split(".")
    for end in range(1, len(parts)):
        package_name = ".".join(parts[:end])
        if package_name not in sys.modules:
            package = types.ModuleType(package_name)
            package.__path__ = []
            monkeypatch.setitem(sys.modules, package_name, package)
    module = types.ModuleType(name)
    for key, value in attributes.items():
        setattr(module, key, value)
    monkeypatch.setitem(sys.modules, name, module)
    if len(parts) > 1:
        parent = sys.modules[".".join(parts[:-1])]
        monkeypatch.setattr(parent, parts[-1], module, raising=False)
    return module


class _CombineInput:
    """Stand-in for ``StandardCombineInput``."""

    def __init__(self, hidden_states):
        self.hidden_states = hidden_states


def _run_bf16_flow(monkeypatch, *, tokens: int, rank: int, two_stream: bool):
    """Run the real dispatch against recording fakes and return what it did."""

    calls = SimpleNamespace(schedule=[], merged=[], allocs=[], moe_kwargs=None)

    class _FakeStream:
        def __init__(self, name):
            self.name = name

        def wait_stream(self, other):
            calls.schedule.append(("wait_stream", self.name, other.name))

    main_stream = _FakeStream("main")
    side_stream = _FakeStream("side")
    stream_state = {"current": main_stream}

    class _StreamContext:
        def __init__(self, stream):
            self.stream = stream
            self.previous = None

        def __enter__(self):
            self.previous = stream_state["current"]
            stream_state["current"] = self.stream

        def __exit__(self, *_args):
            stream_state["current"] = self.previous

    monkeypatch.setattr(torch.cuda, "current_stream", lambda: stream_state["current"])
    monkeypatch.setattr(torch.cuda, "stream", _StreamContext)

    def tracked_new_empty(tensor, *args, **kwargs):
        out = _ORIGINAL_NEW_EMPTY(tensor, *args, **kwargs)
        item = ("alloc", stream_state["current"].name, id(out))
        calls.schedule.append(item)
        calls.allocs.append((stream_state["current"].name, id(out), tuple(out.shape)))
        return out

    monkeypatch.setattr(torch.Tensor, "new_empty", tracked_new_empty)

    def tracked_cat(tensors, *args, **kwargs):
        calls.schedule.append(("cat", stream_state["current"].name))
        return _ORIGINAL_CAT(tensors, *args, **kwargs)

    monkeypatch.setattr(torch, "cat", tracked_cat)

    def merged_experts_fused_moe_lora_add(**kwargs):
        stage = kwargs.get("stage")
        if stage == "routing":
            kind = "routing"
        elif kwargs.get("fuse_sum_all_reduce"):
            kind = "down"
        else:
            kind = "gate_up"
        intermediate = kwargs.get("intermediate_buffer")
        calls.schedule.append(("merged", kind, stream_state["current"].name))
        calls.merged.append(
            {
                "kind": kind,
                "stream": stream_state["current"].name,
                "intermediate_id": None if intermediate is None else id(intermediate),
                # Everything the caller decides, so the single- and two-stream
                # bodies can be compared kwarg by kwarg.
                "flags": {
                    key: value
                    for key, value in kwargs.items()
                    if key != "intermediate_buffer"
                    and not isinstance(value, (torch.Tensor, dict))
                },
            }
        )

    def trtllm_bf16_routed_moe(**kwargs):
        calls.schedule.append(("moe", stream_state["current"].name))
        calls.moe_kwargs = kwargs
        rows = tokens * TOP_K
        return (
            kwargs["output"],
            torch.arange(rows, dtype=torch.int32),
            torch.zeros(rows, INTER),
        )

    def gather_permuted_activation(*_args, **kwargs):
        calls.schedule.append(("gather", stream_state["current"].name))
        return kwargs["out"]

    _stub_module(
        monkeypatch,
        "flashinfer.fused_moe",
        trtllm_bf16_routed_moe=trtllm_bf16_routed_moe,
    )
    _stub_module(
        monkeypatch,
        "sglang.kernels.ops.moe.moe_gather_permuted",
        gather_permuted_activation=gather_permuted_activation,
    )
    _stub_module(
        monkeypatch,
        "sglang.kernels.ops.moe.pack_topk_ids",
        PackTopkIds=SimpleNamespace(
            execute=lambda **_kwargs: torch.zeros(tokens, TOP_K, dtype=torch.int32)
        ),
    )
    _stub_module(
        monkeypatch,
        "sglang.kernels.ops.moe.trtllm_lora_temp.virtual_experts",
        merged_experts_fused_moe_lora_add=merged_experts_fused_moe_lora_add,
    )
    _stub_module(
        monkeypatch,
        "sglang.srt.layers.moe.moe_runner.flashinfer_trtllm",
        fused_experts_none_to_flashinfer_trtllm_bf16=lambda *a, **k: pytest.fail(
            "the no-LoRA fast-out must not fire with an active adapter"
        ),
        get_activation_type=lambda *_a, **_k: 0,
    )
    _stub_module(
        monkeypatch,
        "sglang.srt.layers.moe.token_dispatcher.standard",
        StandardCombineInput=_CombineInput,
    )
    _stub_module(
        monkeypatch,
        "sglang.srt.layers.moe.topk",
        TopKOutputChecker=SimpleNamespace(format_is_standard=lambda _output: True),
    )
    _stub_module(
        monkeypatch,
        "sglang.srt.layers.moe.utils",
        RoutingMethodType=SimpleNamespace(
            Default="default", DeepSeekV3="dsv3", TopK="topk"
        ),
    )
    _stub_module(
        monkeypatch,
        "sglang.srt.model_executor.runner_utils.capture_mode",
        get_is_capture_mode=lambda: False,
    )

    # Module globals the body reaches directly. The symmetric-memory context is a
    # distributed allocator hook with nothing to say about scheduling here.
    monkeypatch.setattr(lora_dispatch, "get_tp_group", lambda: None)
    monkeypatch.setattr(lora_dispatch, "is_allocation_symmetric", lambda: False)
    monkeypatch.setattr(
        lora_dispatch,
        "use_symmetric_memory",
        lambda _group, disabled=False: contextlib.nullcontext(),
    )

    hidden_states = torch.zeros(tokens, HIDDEN)
    dispatch_output = SimpleNamespace(
        hidden_states=hidden_states,
        topk_output=SimpleNamespace(
            topk_ids=torch.zeros(tokens, TOP_K, dtype=torch.int32),
            topk_weights=torch.ones(tokens, TOP_K),
        ),
    )
    quant_info = SimpleNamespace(
        # 4-D so the BlockMajorK assert passes; the fake op never reads it.
        gemm1_weights=torch.zeros(NUM_EXPERTS, 2 * INTER, 1, 1),
        gemm2_weights=torch.zeros(NUM_EXPERTS, HIDDEN, 1, 1),
        global_num_experts=NUM_EXPERTS,
        local_expert_offset=0,
    )
    runner_config = SimpleNamespace(
        activation="silu",
        is_gated=True,
        num_fused_shared_experts=0,
        top_k=TOP_K,
        intermediate_size_per_partition=INTER,
        num_local_experts=NUM_EXPERTS,
        routing_method_type=None,
        routed_scaling_factor=1.0,
    )
    lora_info = SimpleNamespace(
        lora_use_virtual_experts=True,
        max_lora_rank=rank,
        has_active_lora=True,
        experts_shared_outer_loras=False,
        token_lora_mapping=torch.zeros(tokens, dtype=torch.int32),
        gate_up_lora_a_weights=torch.zeros(1, NUM_EXPERTS, 2 * rank, HIDDEN),
        gate_up_lora_b_weights=torch.zeros(1, NUM_EXPERTS, 2 * INTER, rank),
        down_lora_a_weights=torch.zeros(1, NUM_EXPERTS, rank, INTER),
        down_lora_b_weights=torch.zeros(1, NUM_EXPERTS, HIDDEN, rank),
    )

    result = lora_dispatch.fused_experts_none_to_experimental_sgl_trtllm_bf16_lora(
        dispatch_output,
        quant_info,
        runner_config,
        lora_info,
        gate_up_lora_stream=side_stream if two_stream else None,
    )
    assert result.hidden_states.shape == hidden_states.shape
    return calls


def _index(schedule, predicate) -> int:
    index = next(
        (index for index, item in enumerate(schedule) if predicate(item)), None
    )
    assert index is not None, f"no matching entry in schedule: {schedule}"
    return index


@pytest.mark.parametrize("rank", [32, 128], ids=["direct_expand", "generic_expand"])
def test_two_stream_hoists_allocations_and_joins_before_the_delta_is_read(
    monkeypatch, rank
):
    """Reds when the routing pre-warm or the shrink buffer moves onto the side
    stream, when the join is dropped, or when it sinks below the ``torch.cat``
    or the MoE op that read the delta."""

    tokens = 4
    calls = _run_bf16_flow(monkeypatch, tokens=tokens, rank=rank, two_stream=True)

    kinds = [call["kind"] for call in calls.merged]
    assert kinds == ["routing", "gate_up", "down"]

    routing, gate_up, down = calls.merged
    assert routing["stream"] == "main"
    assert gate_up["stream"] == "side"
    assert down["stream"] == "main"

    # The shrink intermediate the side stream writes was allocated on main.
    alloc_streams = {alloc_id: stream for stream, alloc_id, _shape in calls.allocs}
    assert gate_up["intermediate_id"] in alloc_streams
    assert alloc_streams[gate_up["intermediate_id"]] == "main"
    # ... and so was the delta the op consumes.
    delta_allocs = [
        alloc for alloc in calls.allocs if alloc[2] == (tokens, TOP_K, 2 * INTER)
    ]
    assert [alloc[0] for alloc in delta_allocs] == ["main"]

    # Only the rank-specialized expand can swap in the kernel; above 64 the caller
    # copies instead, and must not ask the kernel for a swap it cannot do.
    assert gate_up["flags"]["swap_out_halves"] is (rank <= 64)
    assert gate_up["flags"]["use_direct_expand_add"] is (rank <= 64)

    fork = _index(calls.schedule, lambda item: item == ("wait_stream", "side", "main"))
    side_lora = _index(
        calls.schedule, lambda item: item[:3] == ("merged", "gate_up", "side")
    )
    join = _index(calls.schedule, lambda item: item == ("wait_stream", "main", "side"))
    moe = _index(calls.schedule, lambda item: item[0] == "moe")
    assert fork < side_lora < join < moe

    if rank > 64:
        cat = _index(calls.schedule, lambda item: item[0] == "cat")
        assert join < cat < moe
    else:
        assert not any(item[0] == "cat" for item in calls.schedule)

    # The delta actually reaches the op, and the down-LoRA runs after the gather.
    assert calls.moe_kwargs["gemm1_lora_delta"] is not None
    gather = _index(calls.schedule, lambda item: item[0] == "gather")
    down_index = _index(calls.schedule, lambda item: item[:2] == ("merged", "down"))
    assert moe < gather < down_index


def test_single_stream_forks_nothing_and_asks_for_the_same_lora(monkeypatch):
    """Reds when the stream-less path grows a stream operation, loses the swap,
    or starts pre-warming routing that only the fork needs."""

    single = _run_bf16_flow(monkeypatch, tokens=4, rank=32, two_stream=False)

    assert not any(item[0] in ("wait_stream", "cat") for item in single.schedule)
    assert [call["stream"] for call in single.merged] == ["main", "main"]
    assert [call["kind"] for call in single.merged] == ["gate_up", "down"]
    assert single.merged[0]["intermediate_id"] is None

    two = _run_bf16_flow(monkeypatch, tokens=4, rank=32, two_stream=True)
    two_gate_up = next(call for call in two.merged if call["kind"] == "gate_up")
    # Same LoRA request either way: only where it runs, and whose buffer it
    # accumulates into, may differ.
    assert single.merged[0]["flags"] == two_gate_up["flags"]


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
