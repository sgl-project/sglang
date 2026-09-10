"""Actual FP4 all-gather and BF16 combine with TP=EP=attention-DP=4."""

import itertools
import unittest
from functools import partial
from unittest.mock import patch

import torch
from flashinfer import SfLayout, nvfp4_quantize
from flashinfer.quantization.nvfp4_quantization_utils import (
    current_nvfp4_4over6_config,
    make_nvfp4_global_scale,
)

from sglang.srt.distributed.parallel_state import (
    destroy_distributed_environment,
    destroy_model_parallel,
    get_tp_group,
    init_distributed_environment,
    initialize_model_parallel,
)
from sglang.srt.environ import envs
from sglang.srt.layers.dp_attention import set_dp_buffer_len
from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
from sglang.srt.layers.moe.token_dispatcher.standard import (
    StandardCombineInput,
    StandardDispatcher,
)
from sglang.srt.layers.moe.topk import (
    BypassedTopKOutput,
    StandardTopKOutput,
    TopK,
    TopKConfig,
)
from sglang.srt.layers.moe.utils import (
    MoeA2ABackend,
    MoeRunnerBackend,
    should_use_flashinfer_moe_fp4_allgather,
)
from sglang.srt.layers.quantization.fp4_utils import fp4_quantize
from sglang.srt.runtime_context import get_flags, get_parallel
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler
from sglang.srt.utils.network import get_free_port
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=90, stage="extra-b", runner_config="4-gpu-b200")

_assert_equal = partial(torch.testing.assert_close, rtol=0, atol=0)


def _check_dispatch(
    rank,
    runner,
    per_token,
    sizes=(0, 1, 3, 129),
    quantized=True,
    hidden_size=256,
):
    sizes = list(sizes)
    total = sum(sizes)
    start = sum(sizes[:rank])
    stop = start + sizes[rank]
    num_experts = 8
    torch.manual_seed(42)
    hidden = torch.randn(total, hidden_size, device="cuda", dtype=torch.bfloat16)
    # Different row magnitudes make an incorrectly gathered row scale visible.
    hidden *= torch.linspace(0.02, 2.0, total, device="cuda")[:, None]
    logits = torch.randn(total, num_experts, device="cuda", dtype=torch.bfloat16)
    weights, ids = logits.float().softmax(-1).topk(2)
    ids = ids.to(torch.int32)
    local_hidden = hidden[start:stop]
    if runner == MoeRunnerBackend.FLASHINFER_TRTLLM:
        topk = BypassedTopKOutput(
            hidden_states=local_hidden,
            router_logits=logits[start:stop],
            topk_config=TopKConfig(top_k=2),
        )
    else:
        topk = StandardTopKOutput(weights[start:stop], ids[start:stop], None)
    if sizes[rank] == 0:
        topk = TopK(top_k=2).empty_topk_output(local_hidden.device)
    scale = torch.tensor([2.0], device="cuda", dtype=torch.float32)
    if per_token:
        scale = make_nvfp4_global_scale(
            scale,
            per_token_activation=True,
            nvfp4_4over6_config=current_nvfp4_4over6_config(),
        )
    dispatcher = StandardDispatcher(
        MoeRunnerConfig(
            num_experts=num_experts,
            num_local_experts=num_experts // len(sizes),
            num_fused_shared_experts=0,
        )
    )
    dispatcher.set_quant_config(
        {"input_global_scale": scale, "use_per_token_activation": per_token}
        if quantized
        else {}
    )
    set_dp_buffer_len(total, sizes[rank], dp_max_padding=False, global_num_tokens=sizes)
    group = get_tp_group()
    with patch.object(group, "all_gatherv", wraps=group.all_gatherv) as gather:
        result = dispatcher.dispatch(local_hidden, topk)
    payloads = gather.call_args.args[0]
    if quantized:
        assert all(t.dtype != torch.bfloat16 for t in payloads)
    routing_bytes = num_experts * 4 if isinstance(topk, BypassedTopKOutput) else 16
    activation_bytes = (
        hidden_size // 2 + hidden_size // 16 + 4 * per_token
        if quantized
        else hidden_size * 2
    )
    assert sum(t.numel() * t.element_size() for t in payloads) == sizes[rank] * (
        activation_bytes + routing_bytes
    )
    if quantized:
        assert result.hidden_states.dtype == torch.uint8
        assert result.hidden_states.shape == (total, hidden_size // 2)
        assert result.hidden_states_scale.dtype == torch.float8_e4m3fn
        assert result.hidden_states_scale.shape == (total, hidden_size // 16)
    assert result.hidden_states_pre_quant is None
    if not quantized:
        # Ignored layers must still gather tokens and routing for the EP runner.
        _assert_equal(result.hidden_states, hidden)
        assert result.hidden_states_scale is None
        assert result.hidden_states_per_token_scale is None
    elif total:
        if per_token:
            expected_x, expected_sf, expected_rows = nvfp4_quantize(
                hidden,
                scale,
                sfLayout=SfLayout.layout_linear,
                per_token_activation=True,
                backend="cute-dsl",
            )
            assert result.hidden_states_per_token_scale.dtype == torch.float32
            _assert_equal(
                result.hidden_states_per_token_scale, expected_rows, rtol=0, atol=0
            )
        else:
            expected_x, expected_sf = fp4_quantize(
                hidden, scale, is_sf_swizzled_layout=False
            )
            assert result.hidden_states_per_token_scale is None
        _assert_equal(result.hidden_states, expected_x)
        _assert_equal(
            result.hidden_states_scale.view(torch.uint8),
            expected_sf.view(torch.uint8).reshape(total, hidden_size // 16),
        )
    elif per_token:
        assert result.hidden_states_per_token_scale.shape == (0,)
    if runner == MoeRunnerBackend.FLASHINFER_TRTLLM:
        assert isinstance(result.topk_output, BypassedTopKOutput)
        _assert_equal(result.topk_output.router_logits, logits.float(), rtol=0, atol=0)
    else:
        _assert_equal(result.topk_output.topk_ids, ids)
        _assert_equal(result.topk_output.topk_weights, weights, rtol=0, atol=0)

    # Each expert rank contributes a known partial output. Combine must sum once
    # and return only this attention-DP rank's original token range.
    partial = torch.full_like(hidden, rank + 1)
    combined = dispatcher.combine(StandardCombineInput(partial))
    _assert_equal(
        combined,
        torch.full_like(local_hidden, len(sizes) * (len(sizes) + 1) // 2),
    )
    with envs.SGLANG_MOE_NVFP4_DISPATCH.override(False):
        unquantized = dispatcher.dispatch(local_hidden, topk)
        assert unquantized.hidden_states is local_hidden
        assert unquantized.hidden_states_scale is None
        assert unquantized.hidden_states_per_token_scale is None
        assert unquantized.topk_output is topk


def _worker(rank, world_size, port):
    torch.cuda.set_device(rank)
    set_global_server_args_for_scheduler(ServerArgs(model_path="dummy"))
    init_distributed_environment(
        world_size=world_size,
        rank=rank,
        local_rank=rank,
        distributed_init_method=f"tcp://127.0.0.1:{port}",
    )
    initialize_model_parallel(
        tensor_model_parallel_size=world_size,
        expert_model_parallel_size=world_size,
        attention_data_parallel_size=world_size,
    )
    try:
        with (
            get_parallel().override(
                tp_size=world_size,
                moe_ep_size=world_size,
                moe_ep_rank=rank,
                attn_dp_size=world_size,
                attn_dp_rank=rank,
            ),
            get_flags().dp.override(
                enabled=True,
                buffer_hidden_size=256,
                buffer_dtype=torch.bfloat16,
                buffer_device=torch.device(f"cuda:{rank}"),
            ),
            get_flags().moe.override(
                a2a_backend=MoeA2ABackend.NONE, disable_fp4_allgather=False
            ),
            envs.SGLANG_MOE_NVFP4_DISPATCH.override(True),
            envs.SGLANG_FLASHINFER_CUTEDSL_NVFP4_W4A16.override(False),
        ):
            for runner, (quantization, per_token) in itertools.product(
                (
                    MoeRunnerBackend.FLASHINFER_TRTLLM,
                    MoeRunnerBackend.FLASHINFER_TRTLLM_ROUTED,
                    MoeRunnerBackend.FLASHINFER_CUTEDSL,
                ),
                (
                    ("modelopt_fp4", False),
                    ("modelopt_fp4", True),
                    ("nvfp4_online", True),
                ),
            ):
                with get_flags().moe.override(
                    runner_backend=runner, quantization=quantization
                ):
                    if rank == 0:
                        print(
                            f"Checking {runner.value}: {quantization=}, {per_token=}",
                            flush=True,
                        )
                    assert should_use_flashinfer_moe_fp4_allgather()
                    with get_flags().moe.override(disable_fp4_allgather=True):
                        assert not should_use_flashinfer_moe_fp4_allgather()
                    with get_flags().dp.override(enabled=False):
                        assert not should_use_flashinfer_moe_fp4_allgather()
                    with get_parallel().override(attn_dp_size=2):
                        assert not should_use_flashinfer_moe_fp4_allgather()
                    cases = [{}, {"sizes": [8] * 4}, {"sizes": [0] * 4}]
                    if quantization == "nvfp4_online":
                        # Ignored layers retain BF16; latent experts are narrower
                        # than the model's DP buffer (256).
                        cases += [{"quantized": False}, {"hidden_size": 128}]
                    for case in cases:
                        _check_dispatch(rank, runner, per_token, **case)
                    if runner == MoeRunnerBackend.FLASHINFER_CUTEDSL:
                        with envs.SGLANG_FLASHINFER_CUTEDSL_NVFP4_W4A16.override(True):
                            assert not should_use_flashinfer_moe_fp4_allgather()
    finally:
        destroy_model_parallel()
        destroy_distributed_environment()


class TestStandardFp4Dispatch(unittest.TestCase):
    def test_distributed_dispatch(self):
        torch.multiprocessing.spawn(_worker, args=(4, get_free_port()), nprocs=4)


if __name__ == "__main__":
    unittest.main()
