"""CPU-only runtime-flow tests for the experimental Marlin MoE-LoRA runner:
recording fakes make its kernel ordering, buffer ownership and expert-parallel
index localization observable without CUDA."""

from __future__ import annotations

import ast
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import sglang
from sglang.srt.lora import trtllm_lora_temp
from sglang.srt.lora.marlin_lora_temp import moe_runner
from sglang.srt.lora.trtllm_lora_temp import environ as trtllm_lora_environ
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

_SGLANG_ROOT = Path(sglang.__file__).resolve().parent

# ``torch.Tensor.zero_`` is monkeypatched per flow; hold the pristine method so
# repeated patching inside one session can never chain the recording wrappers.
_ORIGINAL_TENSOR_ZERO = torch.Tensor.zero_


# ---------------------------------------------------------------------------
# Import resolution
# ---------------------------------------------------------------------------


def _module_source_path(dotted_name: str) -> Path | None:
    """Locate an ``sglang.*`` module's source file without importing it."""

    parts = dotted_name.split(".")
    assert parts[0] == "sglang"
    base = _SGLANG_ROOT.joinpath(*parts[1:])
    module_file = base.with_suffix(".py")
    if module_file.is_file():
        return module_file
    package_file = base / "__init__.py"
    if package_file.is_file():
        return package_file
    return None


def _module_level_bindings(path: Path) -> set[str]:
    """Names bound at module scope, including inside module-level branches."""

    names: set[str] = set()

    def walk(body) -> None:
        for node in body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                names.add(node.name)
                continue
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                for alias in node.names:
                    names.add(alias.asname or alias.name.split(".")[0])
                continue
            if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
                for child in ast.walk(node):
                    if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Store):
                        names.add(child.id)
                continue
            # Module-level control flow (if/try/with/for) can bind names too;
            # function and class bodies are deliberately not descended into.
            for attribute in ("body", "orelse", "finalbody"):
                walk(getattr(node, attribute, ()) or ())
            for handler in getattr(node, "handlers", ()):
                walk(handler.body)

    walk(ast.parse(path.read_text()).body)
    return names


def test_cuda_only_kernel_imports_resolve():
    """Guards resolvability of the module-level ``if _is_cuda:`` kernel imports,
    which never run on CPU and are shadowed by this file's fakes; reds when one
    goes stale (#32884 repointed a bad ``moe_sum_reduce_triton`` source)."""

    tree = ast.parse(Path(moe_runner.__file__).read_text())
    guarded = [
        node
        for node in tree.body
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.Name)
        and node.test.id == "_is_cuda"
    ]
    assert len(guarded) == 1, (
        "expected exactly one module-level `if _is_cuda:` kernel-import block in "
        "moe_runner.py; this test can no longer see the imports it guards"
    )

    imports: list[tuple[str, str]] = []
    for node in ast.walk(guarded[0]):
        assert not isinstance(node, ast.Import), (
            "plain `import X` inside the `if _is_cuda:` block is not covered by "
            "this resolution check; extend it"
        )
        if isinstance(node, ast.ImportFrom):
            assert node.level == 0 and node.module, (
                "relative imports inside the `if _is_cuda:` block are not "
                "covered by this resolution check; extend it"
            )
            imports.extend((node.module, alias.name) for alias in node.names)
    assert imports

    third_party = sorted(
        {module for module, _ in imports if not module.startswith("sglang.")}
    )
    assert third_party == ["sgl_kernel"], (
        "a new non-sglang kernel import appeared in the `if _is_cuda:` block "
        f"({third_party}); its symbols cannot be resolved from sglang sources, "
        "so extend this test deliberately"
    )

    unresolved: list[str] = []
    for module, symbol in imports:
        if not module.startswith("sglang."):
            continue
        path = _module_source_path(module)
        if path is None:
            unresolved.append(f"{module} (no such module)")
        elif symbol not in _module_level_bindings(path):
            unresolved.append(f"{module}.{symbol} (not bound at module scope)")
    assert not unresolved, f"unresolvable CUDA-only kernel imports: {unresolved}"


# ---------------------------------------------------------------------------
# Runtime-flow harness
# ---------------------------------------------------------------------------


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


class _DispatchOutput:
    """Stand-in for ``StandardDispatchOutput`` (the runner's isinstance gate)."""

    def __init__(self, hidden_states, topk_output):
        self.hidden_states = hidden_states
        self.topk_output = topk_output


def _run_marlin_flow(
    monkeypatch,
    *,
    tokens: int,
    two_stream: bool = False,
    capture: bool = False,
    active_lora: bool = True,
    base_mapping: bool = False,
    ep: bool = False,
    slots: int = 1,
    rank: int = 1,
):
    """Run the real runner against recording fakes and return what it did
    (scheduling, stream ownership, buffer identity -- never kernel numerics)."""

    calls = SimpleNamespace(
        merged=[],
        fused_tails=0,
        marlin_is_ep=[],
        marlin_output_ids=[],
        align_num_experts=[],
        schedule=[],
        zeroed=[],
        event_records=[],
        event_waits=[],
    )

    class _FakeStream:
        def __init__(self, name):
            self.name = name

        def wait_stream(self, other):
            calls.schedule.append(("wait_stream", self.name, other.name))

        def wait_event(self, event):
            item = ("wait_event", self.name, event.name)
            calls.schedule.append(item)
            calls.event_waits.append(item)

    main_stream = _FakeStream("main")
    side_stream = _FakeStream("side")
    stream_state = {"current": main_stream}
    event_count = 0

    class _FakeEvent:
        def __init__(self):
            nonlocal event_count
            self.name = f"event{event_count}"
            event_count += 1

        def record(self):
            item = ("record", self.name, stream_state["current"].name)
            calls.schedule.append(item)
            calls.event_records.append(item)

    class _StreamContext:
        def __init__(self, stream):
            self.stream = stream
            self.previous = None

        def __enter__(self):
            self.previous = stream_state["current"]
            stream_state["current"] = self.stream

        def __exit__(self, *_args):
            stream_state["current"] = self.previous

    monkeypatch.setattr(torch.cuda, "Event", _FakeEvent)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda: stream_state["current"])
    monkeypatch.setattr(torch.cuda, "stream", _StreamContext)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: capture)

    def tracked_zero(tensor, *args, **kwargs):
        item = ("zero", stream_state["current"].name, id(tensor))
        calls.schedule.append(item)
        calls.zeroed.append(item)
        return _ORIGINAL_TENSOR_ZERO(tensor, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "zero_", tracked_zero)

    def merged_experts_fused_moe_lora_add(**kwargs):
        intermediate = kwargs.get("intermediate_buffer")
        stage = kwargs.get("stage")
        buffer_id = id(intermediate) if intermediate is not None else None
        stream = stream_state["current"].name
        calls.schedule.append(("merged", stage, stream, buffer_id))
        calls.merged.append(
            {
                "stage": stage,
                "stream": stream,
                "intermediate_id": buffer_id,
                "shared_b": kwargs["experts_shared_outer_loras_b"],
                "zero_intermediate": kwargs.get("zero_intermediate", False),
            }
        )

    _stub_module(
        monkeypatch,
        "sglang.srt.layers.moe.token_dispatcher.standard",
        StandardCombineInput=_CombineInput,
        StandardDispatchOutput=_DispatchOutput,
    )
    _stub_module(
        monkeypatch,
        "sglang.kernels.ops.moe.trtllm_lora_temp.virtual_experts",
        merged_experts_fused_moe_lora_add=merged_experts_fused_moe_lora_add,
    )
    _stub_module(
        monkeypatch,
        "sglang.srt.model_executor.runner",
        get_is_capture_mode=lambda: capture,
    )
    monkeypatch.setattr(trtllm_lora_temp, "get_lora_side_stream", lambda: side_stream)
    monkeypatch.setattr(
        trtllm_lora_temp, "is_two_stream_active", lambda _hidden: two_stream
    )
    monkeypatch.setattr(trtllm_lora_environ, "experimental_lora_enabled", lambda: True)

    def fake_align(_topk_ids, _block_size, num_experts, **_kwargs):
        calls.align_num_experts.append(num_experts)
        return (
            torch.zeros(1, dtype=torch.int32),
            torch.zeros(1, dtype=torch.int32),
            torch.ones(1, dtype=torch.int32),
        )

    def fake_marlin_gemm(_x, output, *_args, **kwargs):
        calls.marlin_is_ep.append(kwargs["is_ep"])
        calls.marlin_output_ids.append(id(output))
        calls.schedule.append(
            ("marlin", stream_state["current"].name, len(calls.marlin_is_ep))
        )
        # ``fill_``, never ``zero_``: the zero-fill contracts are asserted by
        # watching ``Tensor.zero_`` calls, so the fakes must not forge one.
        output.fill_(0)
        return output

    def fake_silu_and_mul_add_delta(_x, _delta, output):
        calls.schedule.append(("activation", stream_state["current"].name))
        output.fill_(0)

    def fake_fused_tail(
        routed_base,
        _routed_rank,
        _weights,
        _shared_b,
        output,
        scale,
        *,
        block_m,
        block_k,
    ):
        calls.fused_tails += 1
        calls.schedule.append(
            ("fused_tail", stream_state["current"].name, block_m, block_k)
        )
        output.copy_(routed_base.sum(dim=1) * scale)

    for name, fake in (
        ("moe_align_block_size", fake_align),
        ("marlin_make_workspace", lambda *_args, **_kwargs: None),
        ("get_scalar_type", lambda *_args, **_kwargs: None),
        ("moe_wna16_marlin_gemm", fake_marlin_gemm),
        ("silu_and_mul_add_delta", fake_silu_and_mul_add_delta),
        ("fused_base_shared_lora_reduce", fake_fused_tail),
        ("fused_base_shared_lora_reduce_config", lambda _tokens: (1, 32)),
        # The real gate needs a bf16 SM100 tensor, so CPU can never reach it.
        (
            "_use_fused_shared_outer_tail",
            lambda _info, _hidden, num_tokens, _hidden_size, _topk: num_tokens <= 512,
        ),
    ):
        monkeypatch.setattr(moe_runner, name, fake, raising=False)

    # Per-flow capture-event list: the module-level one would accumulate across
    # cases now that the module is imported once instead of re-loaded per test.
    monkeypatch.setattr(moe_runner, "_MARLIN_LORA_OVERLAP_EVENTS", [])

    hidden_size, num_experts, expert_size, topk = 2, 1, 16, 2
    hidden_states = torch.zeros(tokens, hidden_size)
    topk_ids = torch.zeros(tokens, topk, dtype=torch.int32)
    if ep:
        # Non-local routes arrive already masked out.
        topk_ids[:, 1] = -1
    topk_weights = torch.ones(tokens, topk)
    dispatch_output = _DispatchOutput(
        hidden_states,
        SimpleNamespace(topk_ids=topk_ids, topk_weights=topk_weights),
    )
    quant_info = SimpleNamespace(
        w13_qweight=torch.empty(num_experts, 2),
        w13_bias=None,
        w13_scales=torch.ones(1),
        w13_global_scale=None,
        w13_qzeros=None,
        w13_g_idx=None,
        w13_g_idx_sort_indices=None,
        w2_qweight=torch.empty(num_experts, 1),
        w2_bias=None,
        w2_scales=torch.ones(1),
        w2_global_scale=None,
        w2_qzeros=None,
        w2_g_idx=None,
        w2_g_idx_sort_indices=None,
        weight_bits=4,
        is_k_full=True,
    )
    lora_info = SimpleNamespace(
        lora_use_virtual_experts=True,
        max_lora_rank=rank,
        has_active_lora=active_lora,
        gate_up_lora_a_weights=torch.zeros(slots, 1, 2 * rank, hidden_size),
        gate_up_lora_b_weights=torch.zeros(slots, num_experts, 2 * expert_size, rank),
        down_lora_a_weights=torch.zeros(slots, num_experts, rank, expert_size),
        down_lora_b_weights=torch.zeros(slots, 1, hidden_size, rank),
        token_lora_mapping=(
            torch.full((tokens,), -1, dtype=torch.int32)
            if base_mapping
            else torch.arange(tokens, dtype=torch.int32).remainder(slots)
        ),
        experts_shared_outer_loras=True,
    )
    runner_config = SimpleNamespace(
        activation="silu",
        routed_scaling_factor=1.0,
        # EP: this rank owns `num_experts` of `2 * num_experts` global experts.
        num_experts=2 * num_experts if ep else num_experts,
        num_local_experts=num_experts,
    )

    result = moe_runner.fused_experts_experimental_sgl_marlin_lora(
        dispatch_output, quant_info, runner_config, lora_info
    )
    assert result.hidden_states.shape == hidden_states.shape
    calls.input_ptr = hidden_states.data_ptr()
    calls.result_ptr = result.hidden_states.data_ptr()
    calls.capture_event_count = len(moe_runner._MARLIN_LORA_OVERLAP_EVENTS)
    return calls


def test_ep_uses_local_alignment_and_skips_nonlocal_marlin_blocks(monkeypatch):
    """Guards expert-parallel index localization and both accumulator clears;
    reds when ``moe_align_block_size`` gets the global expert count, ``is_ep``
    is dropped on a Marlin GEMM, or an accumulator is left uninitialized."""

    calls = _run_marlin_flow(monkeypatch, tokens=16, ep=True, slots=2, rank=128)

    assert calls.align_num_experts == [1]
    assert calls.marlin_is_ep == [True, True]

    down_gemm_output = calls.marlin_output_ids[1]
    assert [item[2] for item in calls.zeroed] == [down_gemm_output]
    zero_index = calls.schedule.index(("zero", "main", down_gemm_output))
    down_gemm_index = [
        index for index, item in enumerate(calls.schedule) if item[0] == "marlin"
    ][1]
    assert zero_index < down_gemm_index

    down_stage = [
        call for call in calls.merged if call["stage"] == "all" and call["shared_b"]
    ]
    assert len(down_stage) == 1
    assert down_stage[0]["zero_intermediate"] is True


def test_factored_decode_captured_two_stream_schedule_ownership_and_event_lifetime(
    monkeypatch,
):
    """Guards the captured two-stream decode schedule; reds when LoRA stages gate
    on a live adapter rather than capture-or-adapter, the down-shrink clear/event
    changes stream or order, an event append drops, or the reduce is in place."""

    calls = _run_marlin_flow(
        monkeypatch,
        tokens=16,
        two_stream=True,
        capture=True,
        active_lora=False,
        base_mapping=True,
    )

    # Capture records the LoRA stages even with no live adapter.
    shrinks = [call for call in calls.merged if call["stage"] == "shrink"]
    assert len(shrinks) == 1
    assert calls.fused_tails == 1

    # The shrink runs on the side stream and accumulates into a buffer that was
    # allocated on main, cleared once, on the stream that writes it.
    assert shrinks[0]["stream"] == "side"
    assert [item[1:] for item in calls.zeroed] == [
        ("side", shrinks[0]["intermediate_id"])
    ]

    down_record = calls.event_records[-1]
    down_wait = calls.event_waits[-1]
    assert down_record[2] == "side"
    assert down_wait == ("wait_event", "main", down_record[1])

    zero_index = calls.schedule.index(calls.zeroed[0])
    shrink_index = next(
        index
        for index, item in enumerate(calls.schedule)
        if item[:3] == ("merged", "shrink", "side")
    )
    record_index = calls.schedule.index(down_record)
    wait_index = calls.schedule.index(down_wait)
    fused_index = next(
        index for index, item in enumerate(calls.schedule) if item[0] == "fused_tail"
    )
    assert zero_index < shrink_index < record_index < wait_index < fused_index

    # gate-up delta done, activation done, down shrink done.
    assert calls.capture_event_count == 3

    assert calls.result_ptr != calls.input_ptr


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
