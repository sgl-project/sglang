# SPDX-License-Identifier: Apache-2.0
"""Numeric coverage for the weight-only NVFP4 compressed-tensors MoE scheme.

test_compressed_tensors_fp4_moe_scheme.py stops at weight creation. This drives
the scheme end to end -- checkpoint-format tensors, process_weights_after_loading,
then apply_weights through the FP4 Marlin MoE runner -- and compares against
experts dequantized in PyTorch. That covers what only the scheme can get wrong:
inverting the compressed-tensors global-scale divisor, the [gate; up] order of
w13 that silu_and_mul expects, sizing w13 for non-gated experts, and the Marlin
forward itself.
"""

import sys
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
from sglang.srt.layers.moe.token_dispatcher.standard import StandardDispatchOutput
from sglang.srt.layers.moe.topk import StandardTopKOutput
from sglang.srt.layers.quantization.compressed_tensors.schemes import (
    CompressedTensorsW4A16Nvfp4MoE,
)
from sglang.srt.utils.common import (
    is_sm80_supported,
    is_sm90_supported,
    is_sm100_supported,
    is_sm120_supported,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_marlin_utils import make_nvfp4_weight_and_ref

register_cuda_ci(est_time=8, stage="base-b-kernel-unit", runner_config="1-gpu-small")
register_cuda_ci(est_time=15, stage="base-b-kernel-unit", runner_config="4-gpu-b200")

NUM_EXPERTS = 4
TOP_K = 2
NUM_TOKENS = 17
# NVFP4 Marlin MoE requires hidden_size % 128 == 0 and
# intermediate_size_per_partition % 64 == 0.
HIDDEN = 256
INTERMEDIATE = 128

requires_fp4_marlin = pytest.mark.skipif(
    not (
        is_sm80_supported()
        or is_sm90_supported()
        or is_sm100_supported()
        or is_sm120_supported()
    ),
    reason="Weight-only NVFP4 Marlin MoE requires CUDA SM8X/SM9X/SM100/SM120",
)


def _build_layer(is_gated: bool, activation: str, dtype: torch.dtype):
    """Create the scheme's parameters, fill them the way a checkpoint would,
    and return the dequantized reference experts alongside."""
    config = MoeRunnerConfig(
        num_experts=NUM_EXPERTS,
        num_local_experts=NUM_EXPERTS,
        hidden_size=HIDDEN,
        intermediate_size_per_partition=INTERMEDIATE,
        top_k=TOP_K,
        params_dtype=dtype,
        activation=activation,
        is_gated=is_gated,
    )
    layer = torch.nn.Module()
    layer.moe_runner_config = config
    scheme = CompressedTensorsW4A16Nvfp4MoE()
    scheme.create_weights(
        layer=layer,
        num_experts=NUM_EXPERTS,
        hidden_size=HIDDEN,
        intermediate_size_per_partition=INTERMEDIATE,
        params_dtype=dtype,
        weight_loader=lambda *args, **kwargs: None,
    )
    layer.to("cuda")

    w13_rows = (2 if is_gated else 1) * INTERMEDIATE
    w13_ref = torch.empty(NUM_EXPERTS, w13_rows, HIDDEN, dtype=dtype, device="cuda")
    w2_ref = torch.empty(NUM_EXPERTS, HIDDEN, INTERMEDIATE, dtype=dtype, device="cuda")
    for e in range(NUM_EXPERTS):
        # One matrix for gate and up together, so the two projections share a
        # global scale, as llm-compressor exports them. Each expert gets its
        # own global scale, so a scale applied to the wrong expert shows up.
        fp4, scales, global_scale, ref = make_nvfp4_weight_and_ref(
            w13_rows, HIDDEN, dtype, group_size=16
        )
        layer.w13_weight_packed.data[e].copy_(fp4)
        layer.w13_weight_scale.data[e].copy_(scales)
        # compressed-tensors stores the global scale as a divisor.
        layer.w13_weight_global_scale.data[e].fill_((1 / global_scale).item())
        w13_ref[e] = ref

        fp4, scales, global_scale, ref = make_nvfp4_weight_and_ref(
            HIDDEN, INTERMEDIATE, dtype, group_size=16
        )
        layer.w2_weight_packed.data[e].copy_(fp4)
        layer.w2_weight_scale.data[e].copy_(scales)
        layer.w2_weight_global_scale.data[e] = (1 / global_scale).item()
        w2_ref[e] = ref

    layer.dispatcher = SimpleNamespace(
        local_expert_mapping=None, num_experts=NUM_EXPERTS
    )
    scheme.create_moe_runner(layer, config)
    scheme.process_weights_after_loading(layer)
    return scheme, layer, w13_ref, w2_ref


def _reference(x, w13_ref, w2_ref, topk_weights, topk_ids, is_gated, activation):
    out = torch.zeros(x.shape, dtype=torch.float32, device=x.device)
    for t in range(x.shape[0]):
        for j in range(topk_ids.shape[1]):
            e = topk_ids[t, j].item()
            h = x[t].float() @ w13_ref[e].float().T
            if is_gated:
                gate, up = h.chunk(2)
                act = F.silu(gate) * up
            elif activation == "relu2":
                act = torch.square(F.relu(h))
            else:
                act = F.silu(h)
            out[t] += topk_weights[t, j].float() * (act @ w2_ref[e].float().T)
    return out


@requires_fp4_marlin
@pytest.mark.parametrize(
    "is_gated,activation", [(True, "silu"), (False, "relu2"), (False, "silu")]
)
def test_moe_scheme_matches_dequantized_experts(is_gated, activation):
    torch.manual_seed(0)
    dtype = torch.bfloat16
    scheme, layer, w13_ref, w2_ref = _build_layer(is_gated, activation, dtype)

    x = torch.randn(NUM_TOKENS, HIDDEN, dtype=dtype, device="cuda") / 10
    logits = torch.randn(NUM_TOKENS, NUM_EXPERTS, dtype=torch.float32, device="cuda")
    topk_weights, topk_ids = torch.topk(torch.softmax(logits, dim=-1), TOP_K, dim=-1)
    topk_weights /= topk_weights.sum(dim=-1, keepdim=True)
    topk_ids = topk_ids.to(torch.int32)
    dispatch_output = StandardDispatchOutput(
        x, None, StandardTopKOutput(topk_weights, topk_ids, logits)
    )

    actual = scheme.apply_weights(layer, dispatch_output).hidden_states
    expected = _reference(
        x, w13_ref, w2_ref, topk_weights, topk_ids, is_gated, activation
    )
    torch.cuda.synchronize()

    # Relative, not absolute: the error scales with the output magnitude. The
    # kernel keeps the intermediate activation in bf16 (eps 2**-7 = 0.0078).
    rel = (actual.float() - expected).norm() / expected.norm()
    assert rel < 0.03, f"relative error {rel:.4f} too large"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
