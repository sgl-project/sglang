"""CPU-only runtime-flow tests for experimental_sgl_marlin."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=5, stage="base-b", runner_config="1-gpu-small")

# Skipped on CI: newly-added inkling LoRA test, disabled pending stabilization.
pytestmark = pytest.mark.skip(reason="new inkling LoRA test; disabled on CI")

REPO_ROOT = Path(__file__).resolve().parents[4]
LORA_TEMP_ROOT = REPO_ROOT / "python/sglang/srt/lora"
MARLIN_RUNNER_PATH = LORA_TEMP_ROOT / "marlin_lora_temp/moe_runner.py"
MARLIN_POLICY_PATH = LORA_TEMP_ROOT / "marlin_lora_temp/policy.py"
TWO_STREAM_PATH = LORA_TEMP_ROOT / "trtllm_lora_temp/__init__.py"


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


def _load_file(monkeypatch, name: str, path: Path):
    parts = name.split(".")
    for end in range(1, len(parts)):
        package_name = ".".join(parts[:end])
        if package_name not in sys.modules:
            package = types.ModuleType(package_name)
            package.__path__ = []
            monkeypatch.setitem(sys.modules, package_name, package)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    if len(parts) > 1:
        parent = sys.modules[".".join(parts[:-1])]
        monkeypatch.setattr(parent, parts[-1], module, raising=False)
    return module


def _load_marlin_runner(monkeypatch, name: str):
    _stub_module(monkeypatch, "sglang.srt.utils", is_cuda=lambda: False)
    _load_file(
        monkeypatch,
        "sglang.srt.lora.marlin_lora_temp.policy",
        MARLIN_POLICY_PATH,
    )
    return _load_file(monkeypatch, name, MARLIN_RUNNER_PATH)


@pytest.mark.parametrize(("tokens", "expected"), [(256, True), (257, False)])
def test_two_stream_token_threshold_is_inclusive(monkeypatch, tokens, expected):
    lora_envs = SimpleNamespace(
        SGLANG_TWO_STREAM_MAX_TOKENS=SimpleNamespace(get=lambda: 256)
    )
    _stub_module(monkeypatch, "sglang.srt.environ", envs=SimpleNamespace())
    _stub_module(
        monkeypatch,
        "sglang.srt.lora.trtllm_lora_temp.environ",
        lora_envs=lora_envs,
    )
    module = _load_file(monkeypatch, "_two_stream_under_test", TWO_STREAM_PATH)
    assert module.is_two_stream_active(torch.empty(tokens, 1)) is expected


@pytest.mark.parametrize(
    ("combined_rank", "rank", "expected"),
    [(128, 64, True), (128, 128, False), (256, 64, False)],
)
def test_two_stream_dense_lora_rank_falls_back(
    monkeypatch, combined_rank, rank, expected
):
    lora_envs = SimpleNamespace(
        SGLANG_TWO_STREAM_MAX_TOKENS=SimpleNamespace(get=lambda: 256)
    )
    _stub_module(monkeypatch, "sglang.srt.environ", envs=SimpleNamespace())
    _stub_module(
        monkeypatch,
        "sglang.srt.lora.trtllm_lora_temp.environ",
        lora_envs=lora_envs,
    )
    module = _load_file(monkeypatch, "_two_stream_rank_under_test", TWO_STREAM_PATH)
    assert (
        module.supports_two_stream_dense_lora(
            torch.empty(1, combined_rank, 1), torch.empty(1, 1, rank)
        )
        is expected
    )


class _CombineInput:
    def __init__(self, hidden_states):
        self.hidden_states = hidden_states


class _DispatchOutput:
    def __init__(self, hidden_states, topk_output):
        self.hidden_states = hidden_states
        self.topk_output = topk_output


def _run_marlin_policy(
    monkeypatch,
    *,
    tokens: int,
    master: bool = True,
    two_stream: bool = False,
    capture: bool = False,
    active_lora: bool = True,
    base_mapping: bool = False,
    direct_decode: bool = False,
    ep: bool = False,
    slots: int = 1,
    rank: int = 1,
    shared_outer: bool = True,
    base_value: float = 0.0,
):
    module = _load_marlin_runner(monkeypatch, "_marlin_runner_under_test")
    # The hermetic runner uses tiny CPU tensors. Explicitly emulate the exact
    # B200/Inkling eligibility gate so these tests exercise the fused schedule.
    module._use_fused_shared_outer_tail = (
        lambda _info, _hidden, num_tokens, _hidden_size, _topk: num_tokens <= 512
    )
    module._use_direct_decode_kernels = lambda *_args, **_kwargs: direct_decode

    calls = SimpleNamespace(
        merged=[],
        split_gate_checks=0,
        weighted_rank_sums=0,
        fused_tails=0,
        direct_gate=0,
        direct_down=0,
        marlin_is_ep=[],
        align_num_experts=[],
        cache3_was_zero=None,
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

    original_zero = torch.Tensor.zero_

    def tracked_zero(intermediate, *args, **kwargs):
        item = ("zero", stream_state["current"].name, id(intermediate))
        calls.schedule.append(item)
        calls.zeroed.append(item)
        return original_zero(intermediate, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "zero_", tracked_zero)

    def merged_experts_fused_moe_lora_add(**kwargs):
        intermediate = kwargs.get("intermediate_buffer")
        stage = kwargs.get("stage")
        calls.schedule.append(
            (
                "merged",
                stage,
                stream_state["current"].name,
                id(intermediate) if intermediate is not None else None,
            )
        )
        calls.merged.append(
            {
                "stage": stage,
                "stream": stream_state["current"].name,
                "intermediate_shape": (
                    tuple(kwargs["intermediate_buffer"].shape)
                    if kwargs.get("intermediate_buffer") is not None
                    else None
                ),
                "broadcast": kwargs.get("broadcast_intermediate", False),
                "prewarm_a": kwargs.get("prewarm_a_routing", True),
                "prewarm_b": kwargs.get("prewarm_b_routing", True),
                "topk_shape": tuple(kwargs["topk_ids"].shape),
                "cache_id": id(kwargs.get("routing_cache")),
                "shared_a": kwargs["experts_shared_outer_loras_a"],
                "shared_b": kwargs["experts_shared_outer_loras_b"],
                "fuse_add": kwargs.get("fuse_add_to_output", True),
                "direct_expand": kwargs.get("use_direct_expand_add", False),
                "mul_routed_weight": kwargs["mul_routed_weight"],
                "zero_intermediate": kwargs.get("zero_intermediate", False),
                "mapping": kwargs["token_lora_mapping"].clone(),
                "intermediate_id": (
                    id(intermediate) if intermediate is not None else None
                ),
            }
        )
        if stage == "expand":
            if kwargs.get("fuse_add_to_output", True):
                active = kwargs["token_lora_mapping"] >= 0
                kwargs["output"][active].add_(1)
            else:
                kwargs["output"].fill_(0)
        if stage == "shrink":
            return intermediate

    def is_two_stream_active(_hidden_states):
        calls.split_gate_checks += 1
        return two_stream

    _stub_module(
        monkeypatch,
        "sglang.srt.layers.moe.token_dispatcher.standard",
        StandardCombineInput=_CombineInput,
        StandardDispatchOutput=_DispatchOutput,
    )
    _stub_module(
        monkeypatch,
        "sglang.srt.lora.trtllm_lora_temp",
        get_lora_side_stream=lambda: side_stream,
        is_two_stream_active=is_two_stream_active,
    )
    _stub_module(
        monkeypatch,
        "sglang.srt.lora.trtllm_lora_temp.environ",
        experimental_lora_enabled=lambda: master,
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

    def fake_align(_topk_ids, _block_size, num_experts, **_kwargs):
        calls.align_num_experts.append(num_experts)
        return (
            torch.zeros(1, dtype=torch.int32),
            torch.zeros(1, dtype=torch.int32),
            torch.ones(1, dtype=torch.int32),
        )

    module.moe_align_block_size = fake_align
    module.marlin_make_workspace = lambda *_args, **_kwargs: None
    module.get_scalar_type = lambda *_args, **_kwargs: None

    def fake_marlin_gemm(_x, output, *_args, **_kwargs):
        calls.marlin_is_ep.append(_kwargs["is_ep"])
        if len(calls.marlin_is_ep) == 2:
            calls.cache3_was_zero = bool(torch.count_nonzero(output) == 0)
        calls.schedule.append(
            ("marlin", stream_state["current"].name, len(calls.schedule))
        )
        output.fill_(base_value if len(calls.marlin_is_ep) == 2 else 0)
        return output

    module.moe_wna16_marlin_gemm = fake_marlin_gemm

    def fake_silu_and_mul_add_delta(_x, _delta, output):
        calls.schedule.append(("activation", stream_state["current"].name))
        output.fill_(0)

    module.silu_and_mul_add_delta = fake_silu_and_mul_add_delta
    module.silu_and_mul = lambda _x, output: output.fill_(0)

    def fake_triton_reduce(_input, output, _scale):
        calls.schedule.append(("reduce", stream_state["current"].name))
        output.copy_(_input.sum(dim=1) * _scale)

    module.moe_sum_reduce_triton = fake_triton_reduce

    def fake_weighted_rank_sum(routed_rank, weights, output, scale, *, block_m):
        calls.weighted_rank_sums += 1
        calls.schedule.append(
            ("weighted", stream_state["current"].name, id(routed_rank))
        )
        output.copy_(
            (routed_rank * weights.to(routed_rank.dtype).unsqueeze(-1)).sum(dim=1)
            * scale
        )

    module.weighted_topk_rank_sum = fake_weighted_rank_sum

    def fake_fused_tail(
        routed_base,
        routed_rank,
        weights,
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

    module.fused_base_shared_lora_reduce = fake_fused_tail
    module.fused_base_shared_lora_reduce_config = lambda _tokens: (1, 32)

    def fake_direct_gate(_shared, _weight, _topk_ids, _mapping, output):
        calls.direct_gate += 1
        calls.schedule.append(("direct_gate", stream_state["current"].name))
        output.fill_(0)

    def fake_direct_down(_activation, _weight, _topk_ids, _mapping, output):
        calls.direct_down += 1
        calls.schedule.append(("direct_down", stream_state["current"].name))
        output.fill_(0)

    module.direct_decode_gate_expand = fake_direct_gate
    module.direct_decode_down_shrink = fake_direct_down

    hidden_size, num_experts, expert_size, topk, max_rank = 2, 1, 16, 2, rank
    hidden_states = torch.zeros(tokens, hidden_size)
    topk_ids = torch.zeros(tokens, topk, dtype=torch.int32)
    if ep:
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
        expert_map=None,
        global_num_experts=num_experts,
        weight_bits=4,
        is_k_full=True,
    )
    lora_info = SimpleNamespace(
        lora_use_virtual_experts=True,
        max_lora_rank=max_rank,
        has_active_lora=active_lora,
        gate_up_lora_a_weights=torch.zeros(
            slots,
            1 if shared_outer else num_experts,
            2 * max_rank,
            hidden_size,
        ),
        gate_up_lora_b_weights=torch.zeros(
            slots, num_experts, 2 * expert_size, max_rank
        ),
        down_lora_a_weights=torch.zeros(slots, num_experts, max_rank, expert_size),
        down_lora_b_weights=torch.zeros(
            slots,
            1 if shared_outer else num_experts,
            hidden_size,
            max_rank,
        ),
        token_lora_mapping=(
            torch.full((tokens,), -1, dtype=torch.int32)
            if base_mapping
            else torch.arange(tokens, dtype=torch.int32).remainder(slots)
        ),
        experts_shared_outer_loras=shared_outer,
    )
    runner_config = SimpleNamespace(
        activation="silu",
        routed_scaling_factor=1.0,
        num_experts=num_experts,
        num_local_experts=num_experts,
    )
    if ep:
        runner_config.num_experts = 2 * num_experts

    result = module.fused_experts_experimental_sgl_marlin_lora(
        dispatch_output, quant_info, runner_config, lora_info
    )
    assert result.hidden_states.shape == hidden_states.shape
    calls.input_ptr = hidden_states.data_ptr()
    calls.result_ptr = result.hidden_states.data_ptr()
    calls.result = result.hidden_states
    calls.capture_event_count = len(module._MARLIN_LORA_OVERLAP_EVENTS)
    return calls


@pytest.mark.parametrize(("master", "split_gate_checks"), [(False, 0), (True, 1)])
def test_two_stream_batch_gate_is_master_gated(monkeypatch, master, split_gate_checks):
    calls = _run_marlin_policy(monkeypatch, tokens=1, master=master, two_stream=True)
    assert calls.split_gate_checks == split_gate_checks
    shrinks = [call for call in calls.merged if call["stage"] == "shrink"]
    assert shrinks[0]["stream"] == ("side" if master else "main")


def test_shared_outer_factorization_runtime_flow(monkeypatch):
    tokens = 16
    calls = _run_marlin_policy(monkeypatch, tokens=tokens)
    gate_expand = [
        call for call in calls.merged if call["stage"] == "expand" and call["broadcast"]
    ]
    down_shrink = [call for call in calls.merged if call["stage"] == "shrink"]
    routing = [call for call in calls.merged if call["stage"] == "routing"]

    assert len(gate_expand) == 1
    assert len(down_shrink) == 1
    assert calls.weighted_rank_sums == 0
    assert calls.fused_tails == 1
    assert gate_expand[0]["intermediate_shape"] == (tokens, 2)
    assert down_shrink[0]["intermediate_shape"] == (tokens, 2, 1)
    assert [(call["prewarm_a"], call["prewarm_b"]) for call in routing] == [
        (False, True),
        (True, False),
    ]
    assert down_shrink[0]["prewarm_b"] is False


@pytest.mark.parametrize(
    ("slots", "rank"),
    [(8, 128), (16, 128)],
)
def test_ep_shared_outer_uses_safe_generic_fallback(monkeypatch, slots, rank):
    calls = _run_marlin_policy(
        monkeypatch,
        tokens=1,
        ep=True,
        slots=slots,
        rank=rank,
    )
    generic_down = [
        call for call in calls.merged if call["stage"] == "all" and call["shared_b"]
    ]
    assert len(generic_down) == 1
    assert generic_down[0]["zero_intermediate"] is True
    assert generic_down[0]["direct_expand"] is (rank <= 64)


def test_ep_shared_outer_low_rank_multi_slot_takes_factored_path(monkeypatch):
    # Slot-count gates are lifted: EP shared-outer rank<=64 pools of any size
    # collapse through the factored prefill path instead of the zeroed generic
    # fallback (routing is by adapter slot, so no unowned regions are read).
    calls = _run_marlin_policy(
        monkeypatch,
        tokens=1,
        ep=True,
        slots=5,
        rank=32,
    )
    generic_down = [
        call for call in calls.merged if call["stage"] == "all" and call["shared_b"]
    ]
    assert not generic_down


def test_ep_per_expert_layout_uses_generic_fallback(monkeypatch):
    calls = _run_marlin_policy(
        monkeypatch,
        tokens=1,
        ep=True,
        slots=8,
        rank=128,
        shared_outer=False,
    )
    generic_down = [
        call
        for call in calls.merged
        if call["stage"] == "all" and call["mul_routed_weight"]
    ]
    assert len(generic_down) == 1
    assert generic_down[0]["shared_b"] is False
    assert generic_down[0]["zero_intermediate"] is False
    assert generic_down[0]["direct_expand"] is False


@pytest.mark.parametrize(
    ("case", "expected"),
    [
        ("supported", True),
        ("multi_slot", False),
        ("non_shared", False),
        ("rank_too_large", False),
        ("single_route", False),
        ("empty_batch", False),
        ("mismatched_experts", False),
    ],
)
def test_shared_outer_factorization_eligibility_is_narrow(monkeypatch, case, expected):
    module = _load_marlin_runner(monkeypatch, "_marlin_runner_eligibility")

    rank = 65 if case == "rank_too_large" else 32
    slots = 2 if case == "multi_slot" else 1
    experts = 4
    hidden = 8
    intermediate = 3
    info = SimpleNamespace(
        max_lora_rank=rank,
        experts_shared_outer_loras=case != "non_shared",
        gate_up_lora_a_weights=torch.empty(slots, 1, 2 * rank, hidden),
        gate_up_lora_b_weights=torch.empty(slots, experts, 2 * intermediate, rank),
        down_lora_a_weights=torch.empty(
            slots,
            experts + (1 if case == "mismatched_experts" else 0),
            rank,
            intermediate,
        ),
        down_lora_b_weights=torch.empty(slots, 1, hidden, rank),
    )
    tokens = 0 if case == "empty_batch" else 1
    topk = 1 if case == "single_route" else 2
    assert module._use_shared_outer_factorization(info, tokens, topk) is expected


@pytest.mark.parametrize(
    ("case", "expected"),
    [
        ("supported", True),
        ("three_slots", True),
        ("one_slot", False),
        ("five_slots", True),
        ("sixteen_slots", True),
        ("large_batch", False),
        ("ep", False),
        ("hopper", False),
    ],
)
def test_multi_shared_outer_decode_factorization_is_narrow(monkeypatch, case, expected):
    module = _load_marlin_runner(monkeypatch, "_marlin_runner_multi_policy")
    monkeypatch.setattr(
        torch.cuda,
        "get_device_capability",
        lambda _device: (9, 0) if case == "hopper" else (10, 0),
    )
    slots = (
        1
        if case == "one_slot"
        else (
            3
            if case == "three_slots"
            else 5 if case == "five_slots" else 16 if case == "sixteen_slots" else 4
        )
    )
    info = SimpleNamespace(
        max_lora_rank=32,
        experts_shared_outer_loras=True,
        gate_up_lora_a_weights=SimpleNamespace(shape=(slots, 1, 64, 6144)),
        gate_up_lora_b_weights=SimpleNamespace(shape=(slots, 256, 768, 32)),
        down_lora_a_weights=SimpleNamespace(shape=(slots, 256, 32, 384)),
        down_lora_b_weights=SimpleNamespace(shape=(slots, 1, 6144, 32)),
    )
    hidden_states = SimpleNamespace(
        is_cuda=True, dtype=torch.bfloat16, device=torch.device("cuda")
    )
    assert (
        module._use_multi_shared_outer_decode_factorization(
            info,
            hidden_states,
            num_tokens=33 if case == "large_batch" else 32,
            hidden_size=6144,
            router_topk=6,
            num_experts=256,
            intermediate_size=384,
            ep_active=case == "ep",
        )
        is expected
    )


@pytest.mark.parametrize(
    ("case", "expected"),
    [
        ("supported", True),
        ("four_slots", True),
        ("decode_boundary", False),
        ("one_slot", False),
        ("five_slots", True),
        ("ep", True),
        ("ep_decode", True),
        ("non_shared", False),
        ("rank_too_large", False),
        ("single_route", False),
        ("mismatched_shape", False),
    ],
)
def test_multi_shared_outer_prefill_factorization_is_narrow(
    monkeypatch, case, expected
):
    module = _load_marlin_runner(monkeypatch, "_marlin_runner_prefill_policy")

    slots = (
        1
        if case == "one_slot"
        else 5 if case == "five_slots" else 4 if case == "four_slots" else 2
    )
    rank = 65 if case == "rank_too_large" else 32
    experts = 256
    intermediate = 384
    hidden = 6144
    info = SimpleNamespace(
        max_lora_rank=rank,
        experts_shared_outer_loras=case != "non_shared",
        gate_up_lora_a_weights=SimpleNamespace(shape=(slots, 1, 2 * rank, hidden)),
        gate_up_lora_b_weights=SimpleNamespace(
            shape=(slots, experts, 2 * intermediate, rank)
        ),
        down_lora_a_weights=SimpleNamespace(
            shape=(
                slots,
                experts + (1 if case == "mismatched_shape" else 0),
                rank,
                intermediate,
            )
        ),
        down_lora_b_weights=SimpleNamespace(shape=(slots, 1, hidden, rank)),
    )
    assert (
        module._use_multi_shared_outer_prefill_factorization(
            info,
            num_tokens=32 if case in ("decode_boundary", "ep_decode") else 33,
            hidden_size=hidden,
            router_topk=1 if case == "single_route" else 6,
            num_experts=experts,
            intermediate_size=intermediate,
            ep_active=case in ("ep", "ep_decode"),
        )
        is expected
    )


@pytest.mark.parametrize(
    ("case", "expected"),
    [
        ("supported", True),
        ("unfactored", False),
        ("unfused", False),
        ("ep", False),
        ("large_batch", False),
    ],
)
def test_direct_decode_selection_is_narrow(monkeypatch, case, expected):
    module = _load_marlin_runner(monkeypatch, "_marlin_runner_direct_policy")
    info = SimpleNamespace(
        gate_up_lora_b_weights=SimpleNamespace(shape=(3, 256, 768, 32)),
        down_lora_a_weights=SimpleNamespace(shape=(3, 256, 32, 384)),
    )
    assert (
        module._use_direct_decode_kernels(
            info,
            factored_shared_outer=case != "unfactored",
            fused_shared_outer_tail=case != "unfused",
            ep_active=case == "ep",
            num_tokens=33 if case == "large_batch" else 32,
            num_experts=256,
            intermediate_size=384,
        )
        is expected
    )


@pytest.mark.parametrize(
    ("case", "expected"),
    [
        ("supported", True),
        ("boundary_m", True),
        ("hopper", False),
        ("fp16", False),
        ("rank64", False),
        ("hidden", False),
        ("topk", False),
        ("large_m", False),
    ],
)
def test_fused_shared_outer_tail_is_b200_inkling_specific(monkeypatch, case, expected):
    module = _load_marlin_runner(monkeypatch, "_marlin_runner_tail_policy")
    monkeypatch.setattr(
        torch.cuda,
        "get_device_capability",
        lambda _device: (9, 0) if case == "hopper" else (10, 0),
    )
    info = SimpleNamespace(max_lora_rank=64 if case == "rank64" else 32)
    hidden_states = SimpleNamespace(
        is_cuda=True,
        dtype=torch.float16 if case == "fp16" else torch.bfloat16,
        device=torch.device("cuda"),
    )
    assert (
        module._use_fused_shared_outer_tail(
            info,
            hidden_states,
            513 if case == "large_m" else 512 if case == "boundary_m" else 32,
            4096 if case == "hidden" else 6144,
            8 if case == "topk" else 6,
        )
        is expected
    )


def test_multi_prefill_collapses_only_shared_factors_and_separates_caches(
    monkeypatch,
):
    calls = _run_marlin_policy(monkeypatch, tokens=64, slots=3)

    routing = [call for call in calls.merged if call["stage"] == "routing"]
    full_routing = [call for call in routing if call["topk_shape"] == (64, 2)]
    collapsed_routing = [call for call in routing if call["topk_shape"] == (64, 1)]
    assert [(call["prewarm_a"], call["prewarm_b"]) for call in full_routing] == [
        (False, True),  # real-route per-expert gate B
        (True, False),  # real-route per-expert down A
    ]
    assert [(call["prewarm_a"], call["prewarm_b"]) for call in collapsed_routing] == [
        (True, False),  # collapsed selected shared gate A
        (False, True),  # collapsed selected shared down B
    ]
    assert len({call["cache_id"] for call in full_routing}) == 1
    assert len({call["cache_id"] for call in collapsed_routing}) == 1
    assert full_routing[0]["cache_id"] != collapsed_routing[0]["cache_id"]

    gate_shrink = next(
        call for call in calls.merged if call["stage"] == "shrink" and call["shared_a"]
    )
    gate_expand = next(
        call for call in calls.merged if call["stage"] == "expand" and call["broadcast"]
    )
    down_shrink = next(
        call
        for call in calls.merged
        if call["stage"] == "shrink" and not call["shared_a"]
    )
    down_expand = next(
        call for call in calls.merged if call["stage"] == "expand" and call["shared_b"]
    )

    assert gate_shrink["topk_shape"] == (64, 1)
    assert gate_shrink["intermediate_shape"] == (64, 2)
    assert gate_expand["topk_shape"] == (64, 2)
    assert gate_expand["intermediate_shape"] == (64, 2)
    assert down_shrink["topk_shape"] == (64, 2)
    assert down_shrink["intermediate_shape"] == (64, 2, 1)
    assert down_expand["topk_shape"] == (64, 1)
    assert down_expand["intermediate_shape"] == (64, 1)
    assert down_expand["fuse_add"] is True
    assert down_expand["direct_expand"] is False
    assert down_expand["mul_routed_weight"] is False
    assert calls.weighted_rank_sums == 1
    # The mapped one-token-per-CTA tail remains decode-only.
    assert calls.fused_tails == 0


def test_multi_prefill_none_rows_preserve_base_reduction(monkeypatch):
    calls = _run_marlin_policy(
        monkeypatch,
        tokens=64,
        slots=2,
        capture=True,
        active_lora=False,
        base_mapping=True,
        base_value=3.0,
    )

    down_expand = next(
        call for call in calls.merged if call["stage"] == "expand" and call["shared_b"]
    )
    assert down_expand["topk_shape"] == (64, 1)
    assert torch.equal(down_expand["mapping"], torch.full((64,), -1, dtype=torch.int32))
    # The fake Marlin down output is 3 for each of two routes. The collapsed
    # shared-B expand masks every None row, so it must leave the base sum at 6.
    torch.testing.assert_close(calls.result, torch.full_like(calls.result, 6.0))
    assert calls.fused_tails == 0


def test_direct_decode_skips_virtual_routing_and_zero_fill(monkeypatch):
    calls = _run_marlin_policy(
        monkeypatch, tokens=16, two_stream=True, direct_decode=True
    )

    assert [call for call in calls.merged if call["stage"] == "routing"] == []
    assert [call for call in calls.merged if call["stage"] == "shrink"] == []
    assert calls.direct_gate == 1
    assert calls.direct_down == 1
    assert calls.zeroed == []
    assert calls.fused_tails == 1


def test_ep_uses_local_alignment_and_skips_nonlocal_marlin_blocks(monkeypatch):
    calls = _run_marlin_policy(monkeypatch, tokens=16, ep=True)

    assert calls.align_num_experts == [1]
    assert calls.marlin_is_ep == [True, True]
    assert calls.cache3_was_zero is True


def test_factored_decode_two_stream_schedule_and_ownership(monkeypatch):
    calls = _run_marlin_policy(monkeypatch, tokens=16, two_stream=True)
    shrinks = [call for call in calls.merged if call["stage"] == "shrink"]

    assert len(shrinks) == 1
    assert shrinks[0]["stream"] == "side"
    assert shrinks[0]["prewarm_b"] is False
    assert len(calls.zeroed) == 1
    assert calls.zeroed[0][1] == "side"

    buffer_id = shrinks[0]["intermediate_id"]
    assert calls.zeroed[0][2] == buffer_id
    zero_index = calls.schedule.index(calls.zeroed[0])
    shrink_index = next(
        index
        for index, item in enumerate(calls.schedule)
        if item[:3] == ("merged", "shrink", "side")
    )
    down_record = calls.event_records[-1]
    down_wait = calls.event_waits[-1]
    record_index = calls.schedule.index(down_record)
    wait_index = calls.schedule.index(down_wait)
    fused_index = next(
        index for index, item in enumerate(calls.schedule) if item[0] == "fused_tail"
    )
    assert down_record[2] == "side"
    assert down_wait == ("wait_event", "main", down_record[1])
    assert zero_index < shrink_index < record_index < wait_index < fused_index
    assert fused_index == len(calls.schedule) - 1


def test_factored_decode_single_stream_fallback_has_one_main_shrink(monkeypatch):
    calls = _run_marlin_policy(monkeypatch, tokens=16, two_stream=False)
    shrinks = [call for call in calls.merged if call["stage"] == "shrink"]

    assert len(shrinks) == 1
    assert shrinks[0]["stream"] == "main"
    assert shrinks[0]["prewarm_b"] is False
    assert len(calls.zeroed) == 1
    assert calls.zeroed[0][1] == "main"
    assert calls.event_records == []
    assert calls.event_waits == []

    buffer_id = shrinks[0]["intermediate_id"]
    assert calls.zeroed[0][2] == buffer_id
    second_marlin_index = max(
        index for index, item in enumerate(calls.schedule) if item[0] == "marlin"
    )
    zero_index = calls.schedule.index(calls.zeroed[0])
    shrink_index = next(
        index
        for index, item in enumerate(calls.schedule)
        if item[:3] == ("merged", "shrink", "main")
    )
    fused_index = next(
        index for index, item in enumerate(calls.schedule) if item[0] == "fused_tail"
    )
    assert second_marlin_index < zero_index < shrink_index < fused_index


def test_factored_decode_capture_base_rows_keep_main_owned_buffers(monkeypatch):
    calls = _run_marlin_policy(
        monkeypatch,
        tokens=16,
        two_stream=True,
        capture=True,
        active_lora=False,
        base_mapping=True,
    )
    shrinks = [call for call in calls.merged if call["stage"] == "shrink"]

    assert len(shrinks) == 1
    assert len(calls.zeroed) == 1
    assert calls.zeroed[0][1] == "side"
    assert calls.zeroed[0][2] == shrinks[0]["intermediate_id"]
    assert calls.weighted_rank_sums == 0
    assert calls.fused_tails == 1
    assert calls.capture_event_count == 3
    assert calls.result_ptr != calls.input_ptr
    torch.testing.assert_close(calls.result, torch.zeros_like(calls.result))


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
