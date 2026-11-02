import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import call, patch

import pytest
import torch

from sglang.srt.layers import zero_copy_context
from sglang.srt.layers.moe.token_dispatcher.standard import StandardDispatchOutput
from sglang.srt.layers.moe.topk import StandardTopKOutput
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


@pytest.mark.parametrize(
    "tokens,precision,packed,defer_finalize,launch_limit",
    [
        pytest.param(16387, "default", True, False, 0, id="disabled-limit"),
        pytest.param(16387, "default", True, False, -1, id="negative-limit"),
        pytest.param(3, "default", True, False, 8192, id="single-launch"),
        pytest.param(8192, "default", True, False, 8192, id="exact-launch-limit"),
        pytest.param(16384, "default", True, False, 8192, id="two-full-chunks"),
        pytest.param(16387, "default", True, False, 8192, id="packed-tail-chunk"),
        pytest.param(16387, "default", False, False, 8192, id="tuple-tail-chunk"),
        pytest.param(16387, "bf16", False, False, 8192, id="no-scale-tail-chunk"),
        pytest.param(
            16387, "default", True, True, 8192, id="packed-deferred-one-launch"
        ),
        pytest.param(
            16387, "default", False, True, 8192, id="tuple-deferred-one-launch"
        ),
        pytest.param(
            16387, "bf16", False, True, 8192, id="no-scale-deferred-one-launch"
        ),
    ],
)
def test_situ_routed_moe_returns_published_output_buffer(
    tokens, precision, packed, defer_finalize, launch_limit
):
    # This checks launch slicing and the explicit output ABI on CPU. The fake
    # kernel makes unwritten/duplicated rows deterministic instead of depending
    # on uninitialized CUDA memory or requiring real MoE weights.
    # Import mxfp4 before flashinfer_trtllm to avoid the pre-existing
    # compressed_tensors circular import.
    # isort: off
    from sglang.srt.layers.quantization import mxfp4 as mxfp4_module
    from sglang.srt.layers.moe.moe_runner import (
        flashinfer_trtllm as flashinfer_trtllm_module,
    )
    from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
    from sglang.srt.layers.moe.moe_runner.flashinfer_trtllm import (
        FlashInferTrtllmDeferredFinalizeOutput,
        FlashInferTrtllmGenMxfp4MoeQuantInfo,
        _fused_experts_flashinfer_mxfp4_sm100_trtllm_gen,
        flashinfer_trtllm_deferred_finalize_context,
    )
    # isort: on

    hidden, top_k = 128, 2
    x = torch.zeros(tokens, hidden, dtype=torch.bfloat16)
    x_quant = (
        torch.arange(tokens)
        .remainder(251)
        .to(torch.uint8)[:, None]
        .expand(tokens, hidden)
        .contiguous()
    )
    x_scale = (
        torch.arange(tokens)
        .remainder(127)
        .to(torch.uint8)[:, None]
        .expand(tokens, hidden // 32)
        .contiguous()
    )
    if precision == "bf16":
        x_quant, x_scale = x, None
    packed_topk = torch.arange(tokens * top_k, dtype=torch.int32).reshape(tokens, top_k)
    topk_output = StandardTopKOutput(
        topk_weights=torch.arange(tokens * top_k, dtype=torch.float32).reshape(
            tokens, top_k
        ),
        topk_ids=packed_topk.clone(),
        router_logits=torch.empty(tokens, 0, dtype=torch.float32),
    )
    dispatch_output = StandardDispatchOutput(
        hidden_states=x,
        hidden_states_scale=None,
        topk_output=topk_output,
    )

    dummy = torch.empty(1, dtype=torch.uint8)
    quant_info = FlashInferTrtllmGenMxfp4MoeQuantInfo(
        w13_weight=dummy,
        w2_weight=dummy,
        w13_weight_scale=dummy,
        w2_weight_scale=dummy,
        w13_weight_bias=dummy,
        w2_weight_bias=dummy,
        gemm1_alpha=dummy,
        gemm1_beta=dummy,
        gemm1_clamp_limit=dummy,
        global_num_experts=1,
        local_expert_offset=0,
        local_num_experts=1,
        intermediate_size_per_partition=128,
        hidden_size=hidden,
        flashinfer_mxfp4_moe_precision=precision,
    )
    expected = (
        torch.arange(tokens * hidden, dtype=torch.float32)
        .reshape(tokens, hidden)
        .to(torch.bfloat16)
    )
    latent = torch.full_like(x, float("nan"))
    write_counts = torch.zeros(tokens, dtype=torch.int32)
    launches = []
    returned_tensors = []
    deferred_result = (
        expected,
        topk_output.topk_weights,
        torch.zeros(tokens, top_k, dtype=torch.int32),
    )

    def fake_routed_moe(**kwargs):
        actual_x = kwargs["hidden_states"]
        start = actual_x.storage_offset() // x_quant.stride(0)
        end = start + actual_x.shape[0]
        launches.append((start, end))
        torch.testing.assert_close(actual_x, x_quant[start:end], rtol=0, atol=0)
        actual_routing = kwargs["topk_ids"]
        if packed:
            torch.testing.assert_close(
                actual_routing, packed_topk[start:end], rtol=0, atol=0
            )
        else:
            assert isinstance(actual_routing, tuple)
            torch.testing.assert_close(
                actual_routing[0], topk_output.topk_ids[start:end], rtol=0, atol=0
            )
            torch.testing.assert_close(
                actual_routing[1], topk_output.topk_weights[start:end], rtol=0, atol=0
            )
        if x_scale is None:
            assert kwargs["hidden_states_scale"] is None
        else:
            torch.testing.assert_close(
                kwargs["hidden_states_scale"], x_scale[start:end], rtol=0, atol=0
            )
        assert kwargs["output"].data_ptr() == latent[start:end].data_ptr()
        assert kwargs["output"].shape == (end - start, hidden)
        assert kwargs["top_k"] == top_k
        assert kwargs["do_finalize"] is not defer_finalize
        assert kwargs["tune_max_num_tokens"] == 1 << (end - start - 1).bit_length()
        assert kwargs["enable_pdl"] is False
        write_counts[start:end] += 1
        kwargs["output"].copy_(expected[start:end])
        if defer_finalize:
            return deferred_result
        ffi_result = kwargs["output"].clone()
        returned_tensors.append(ffi_result)
        return ffi_result

    flashinfer = ModuleType("flashinfer")
    flashinfer.__path__ = []
    flashinfer.trtllm_fp4_block_scale_moe = None
    fused_moe = ModuleType("flashinfer.fused_moe")
    fused_moe.trtllm_fp4_block_scale_routed_moe = fake_routed_moe
    tllm_enums = ModuleType("flashinfer.tllm_enums")
    tllm_enums.RoutingMethodType = SimpleNamespace(TopK=SimpleNamespace(value=0))
    tllm_enums.ActivationType = SimpleNamespace(Situ=SimpleNamespace(value=0))

    with (
        patch.object(
            flashinfer_trtllm_module,
            "_MXFP4_MOE_MAX_TOKENS_PER_LAUNCH",
            launch_limit,
        ),
        patch.object(
            mxfp4_module,
            "_prepare_flashinfer_mxfp8_activations",
            return_value=(x, packed_topk if packed else None, x_quant, x_scale),
        ),
        patch.dict(
            "sys.modules",
            {
                "flashinfer": flashinfer,
                "flashinfer.fused_moe": fused_moe,
                "flashinfer.tllm_enums": tllm_enums,
            },
        ),
        patch.object(
            flashinfer_trtllm_module, "trtllm_moe_enable_pdl", return_value=False
        ) as enable_pdl,
        zero_copy_context.set_moe_output(latent),
        flashinfer_trtllm_deferred_finalize_context(defer_finalize),
    ):
        combine_input = _fused_experts_flashinfer_mxfp4_sm100_trtllm_gen(
            dispatch_output,
            quant_info,
            MoeRunnerConfig(activation="situ"),
        )

    expected_launches = (
        [(0, tokens)]
        if defer_finalize or launch_limit <= 0
        else [
            (start, min(start + launch_limit, tokens))
            for start in range(0, tokens, launch_limit)
        ]
    )
    assert launches == expected_launches
    assert torch.equal(write_counts, torch.ones_like(write_counts))
    assert enable_pdl.call_args_list == [
        call(end - start) for start, end in expected_launches
    ]
    if defer_finalize:
        deferred_output = combine_input.hidden_states
        assert isinstance(deferred_output, FlashInferTrtllmDeferredFinalizeOutput)
        assert deferred_output.gemm2_out is deferred_result[0]
        assert deferred_output.expert_weights is deferred_result[1]
        assert deferred_output.expanded_idx_to_permuted_idx is deferred_result[2]
        assert deferred_output.top_k == top_k
    else:
        assert len(returned_tensors) == len(expected_launches)
        assert all(
            tensor.data_ptr() != latent.data_ptr() for tensor in returned_tensors
        )
        assert combine_input.hidden_states.data_ptr() == latent.data_ptr()
        torch.testing.assert_close(
            combine_input.hidden_states, expected, rtol=0, atol=0
        )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
