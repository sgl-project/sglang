"""Golden-comparison test for kernels.ops.moe.trtllm_gen_moe (TRT-LLM-gen SiTU MoE).

Needs on-box internal artifacts, so it self-skips when they are absent:

  * ``SGLANG_TRTLLM_GEN_MOE_SDK`` -> private FlashInfer snapshot checkout
    (csrc/, include/, 3rdparty/, local_cubins/ with the SiTU pool);
  * ``SGLANG_TRTLLM_GEN_MOE_GOLDEN`` -> .pt of inputs/shuffled-weights/outputs
    produced by the fork's own harness in a separate process (the fork python
    cannot be imported next to the serving flashinfer).

Validated expectation (GB300, 2026-07-18): outputs are bit-exact against the
fork kernel for w4a16 and w4a8 at T=1 and T=64 (same cubins, same config
selection); the tolerance below only guards against a future tactic split.
"""

import os

import pytest
import torch

from sglang.kernels.ops.moe import trtllm_gen_moe
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=120, stage="base-b-kernel-unit", runner_config="1-gpu-large")

NUM_EXPERTS = 896
TOP_K = 16
INTERMEDIATE = 384

_GOLDEN = os.environ.get("SGLANG_TRTLLM_GEN_MOE_GOLDEN", "")

pytestmark = pytest.mark.skipif(
    not (trtllm_gen_moe.available() and os.path.isfile(_GOLDEN)),
    reason="needs SGLANG_TRTLLM_GEN_MOE_SDK + SGLANG_TRTLLM_GEN_MOE_GOLDEN "
    "(internal on-box artifacts)",
)


@pytest.fixture(scope="module")
def golden():
    return torch.load(_GOLDEN)


@pytest.mark.parametrize("mode", ["w4a16", "w4a8"])
@pytest.mark.parametrize("tokens", [1, 64])
def test_matches_fork_kernel(golden, mode, tokens):
    data = golden[mode]
    case = data["cases"][tokens]
    hidden_states = case["hidden_states"].cuda()
    hidden_states_scale = None
    if mode == "w4a8":
        from flashinfer import mxfp8_quantize

        hidden_states, hidden_states_scale = mxfp8_quantize(hidden_states, False)
        hidden_states_scale = hidden_states_scale.view(torch.float8_e4m3fn).reshape(
            tokens, -1
        )
    alpha = torch.full((NUM_EXPERTS,), 4.0, dtype=torch.float32, device="cuda")
    beta = torch.full((NUM_EXPERTS,), 25.0, dtype=torch.float32, device="cuda")
    out = trtllm_gen_moe.trtllm_fp4_block_scale_moe(
        routing_logits=case["logits"].cuda(),
        routing_bias=data["bias"].cuda(),
        hidden_states=hidden_states,
        hidden_states_scale=hidden_states_scale,
        gemm1_weights=data["gemm1_weights"].cuda(),
        gemm1_weights_scale=data["gemm1_scales"].cuda(),
        gemm1_alpha=alpha,
        gemm1_beta=beta,
        gemm2_weights=data["gemm2_weights"].cuda(),
        gemm2_weights_scale=data["gemm2_scales"].cuda(),
        output1_scale_scalar=data["scale_c_fc1"].cuda(),
        output1_scale_gate_scalar=data["scale_gate_fc1"].cuda(),
        output2_scale_scalar=data["scale_c_fc2"].cuda(),
        num_experts=NUM_EXPERTS,
        top_k=TOP_K,
        n_group=1,
        topk_group=1,
        intermediate_size=INTERMEDIATE,
        routed_scaling_factor=1.0,
    )
    ref = case["output"].cuda().to(torch.float32)
    diff = (out.to(torch.float32) - ref).abs()
    assert diff.max().item() < 0.2 and diff.mean().item() < 0.02
