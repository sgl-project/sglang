"""Focused SM100 trtllm-gen MXFP4 MoE regression test.

``Mxfp4MoEMethod.apply`` (SM100 branch, via the unified MoeRunner) must feed
``trtllm_fp4_block_scale_moe`` the same args a direct kernel call does, so the
two outputs stay bit-exact.

Fixtures are raw checkpoint-order MXFP4, converted by the production
``process_weights_after_loading`` so the kernel sees the interleaved weights and
float8_e4m3fn block scales it gets in a real run.
"""

from __future__ import annotations

from contextlib import nullcontext

import pytest
import torch
from flashinfer import trtllm_fp4_block_scale_moe

from sglang.srt.utils import is_sm100_supported
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=120, stage="base-b", runner_config="4-gpu-b200")

if not is_sm100_supported():
    pytest.skip(
        reason="trtllm-gen MXFP4 requires SM100 (Blackwell).",
        allow_module_level=True,
    )

GROUP_SIZE = 32  # MXFP4 block size


class _MockLayer:
    """Hand-built ``FusedMoE`` stand-in (avoids distributed init)."""


def _make_random_mxfp4(num_experts, hidden, inter, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    w13 = torch.randint(
        0,
        256,
        (num_experts, 2 * inter, hidden // 2),
        dtype=torch.uint8,
        device="cuda",
        generator=g,
    )
    w2 = torch.randint(
        0,
        256,
        (num_experts, hidden, inter // 2),
        dtype=torch.uint8,
        device="cuda",
        generator=g,
    )
    # E8M0 scales centered around 127 (= 2^0); narrow band keeps dequant values
    # sane so the SwiGLU clamp doesn't dominate.
    w13_s = torch.randint(
        125,
        130,
        (num_experts, 2 * inter, hidden // GROUP_SIZE),
        dtype=torch.uint8,
        device="cuda",
        generator=g,
    )
    w2_s = torch.randint(
        125,
        130,
        (num_experts, hidden, inter // GROUP_SIZE),
        dtype=torch.uint8,
        device="cuda",
        generator=g,
    )
    w13_b = (
        torch.randn(
            num_experts, 2 * inter, dtype=torch.float32, device="cuda", generator=g
        )
        * 0.01
    )
    w2_b = (
        torch.randn(
            num_experts, hidden, dtype=torch.float32, device="cuda", generator=g
        )
        * 0.01
    )
    return w13, w2, w13_s, w2_s, w13_b, w2_b


def _build_mock_layer(num_experts, hidden, inter, fixtures):
    """Raw checkpoint-order weights; ``process_weights_after_loading`` converts
    them in place and seeds the SwiGLU scalars, so nothing is pre-applied here."""
    w13, w2, w13_s, w2_s, w13_b, w2_b = fixtures
    layer = _MockLayer()
    layer.w13_weight = torch.nn.Parameter(w13.clone(), requires_grad=False)
    layer.w2_weight = torch.nn.Parameter(w2.clone(), requires_grad=False)
    layer.w13_weight_scale = torch.nn.Parameter(w13_s.clone(), requires_grad=False)
    layer.w2_weight_scale = torch.nn.Parameter(w2_s.clone(), requires_grad=False)
    layer.w13_weight_bias = torch.nn.Parameter(w13_b.clone(), requires_grad=False)
    layer.w2_weight_bias = torch.nn.Parameter(w2_b.clone(), requires_grad=False)
    layer.num_experts = num_experts
    layer.num_local_experts = num_experts  # tests run with EP size = 1
    layer.moe_ep_rank = 0
    return layer


def _build_method(num_experts, hidden, inter, precision):
    from sglang.srt.layers.quantization.mxfp4 import Mxfp4MoEMethod

    method = Mxfp4MoEMethod.__new__(Mxfp4MoEMethod)
    method._fi_kernel = "trtllm_sm100"
    method.use_flashinfer = True
    method.use_marlin = False
    method.use_deep_gemm = False
    method.use_mega_moe = False
    method.num_experts = num_experts
    method.hidden_size = hidden
    method.intermediate_size_per_partition = inter
    method.flashinfer_mxfp4_moe_precision = precision
    method.runner = _build_flashinfer_mxfp4_runner(num_experts, hidden, inter)
    method.moe_runner_config = method.runner.config
    return method


def _build_flashinfer_mxfp4_runner(num_experts, hidden, inter):
    # Bypass create_moe_runner (needs a live server arg context); the fused func
    # only reads dispatch_output / quant_info, so a minimal config suffices.
    import sglang.srt.layers.moe.moe_runner.flashinfer_cutlass  # noqa: F401
    from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
    from sglang.srt.layers.moe.moe_runner.runner import MoeRunner
    from sglang.srt.layers.moe.utils import MoeRunnerBackend

    cfg = MoeRunnerConfig(
        num_experts=num_experts,
        num_local_experts=num_experts,
        hidden_size=hidden,
        intermediate_size_per_partition=inter,
        top_k=None,
        activation="silu",
        is_gated=True,
    )
    return MoeRunner(MoeRunnerBackend.FLASHINFER_MXFP4, cfg)


@pytest.mark.parametrize("use_mega_moe", [False, True])
def test_create_moe_runner_handles_flashinfer_for_megamoe(monkeypatch, use_mega_moe):
    import sglang.srt.layers.quantization.mxfp4 as mxfp4_mod
    from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
    from sglang.srt.layers.moe.utils import MoeRunnerBackend

    runner = object()

    def build_runner(backend, config):
        assert not use_mega_moe
        assert backend == MoeRunnerBackend.FLASHINFER_MXFP4
        assert config is runner_config
        return runner

    monkeypatch.setattr(
        mxfp4_mod,
        "get_moe_runner_backend",
        lambda: MoeRunnerBackend.FLASHINFER_MXFP4,
    )
    monkeypatch.setattr(mxfp4_mod, "MoeRunner", build_runner)

    method = mxfp4_mod.Mxfp4MoEMethod.__new__(mxfp4_mod.Mxfp4MoEMethod)
    method._fi_kernel = "trtllm_sm100"
    method.use_mega_moe = use_mega_moe
    runner_config = MoeRunnerConfig()

    method.create_moe_runner(object(), runner_config)

    assert method.moe_runner_config is runner_config
    if use_mega_moe:
        # FusedMoEMethodBase declares ``runner: MoeRunner | None = None``, so the
        # early return leaves the class default rather than no attribute at all.
        assert method.runner is None
    else:
        assert method.runner is runner


class _MockDispatchOutput:
    # SM100 keeps BYPASSED topk (kernel routes from router_logits), so the
    # dispatch output must carry a real BypassedTopKOutput.
    def __init__(self, hidden_states, router_logits, top_k):
        from sglang.srt.layers.moe.topk import BypassedTopKOutput, TopKConfig

        self.hidden_states = hidden_states
        self.topk_output = BypassedTopKOutput(
            hidden_states=hidden_states,
            router_logits=router_logits,
            topk_config=TopKConfig(top_k=top_k, renormalize=True),
        )


def _quant_input(x, precision, hidden_size):
    # Mirror the SM100 helper's input-quant branch so the reference feeds the
    # kernel the same x_quant / x_scale the SGLang path does.
    origin = x.shape[-1]
    if precision == "bf16":
        x_quant = x
        x_scale = None
        if hidden_size != origin:
            x_quant = torch.nn.functional.pad(
                x_quant, (0, hidden_size - origin), mode="constant", value=0.0
            )
    elif precision == "default":
        if x.shape[-1] == hidden_size:
            if x.dim() > 2:
                x = x.view(-1, x.shape[-1])
            from sglang.kernels.ops.quantization.per_token_group_quant import (
                per_token_group_quant,
            )

            x_quant, x_scale = per_token_group_quant(x, group_size=32, scale_ue8m0=True)
            x_scale = x_scale.view(torch.float8_e4m3fn)
        else:
            from sglang.srt.layers.quantization.fp8_utils import (
                flashinfer_mxfp8_quantize,
            )

            x_quant, x_scale = flashinfer_mxfp8_quantize(
                x, False, alignment=hidden_size
            )
            x_scale = x_scale.view(torch.float8_e4m3fn).reshape(*x.shape[:-1], -1)
    else:
        raise AssertionError(precision)
    return x_quant, x_scale


def _ref_trtllm(x, layer, method, precision, top_k, router_logits):
    # Direct kernel call mirroring the SM100 helper's arg list.
    from sglang.srt.layers.moe.moe_runner.flashinfer_trtllm import (
        trtllm_moe_enable_pdl,
    )
    from sglang.srt.utils.common import next_power_of_2

    x_quant, x_scale = _quant_input(x, precision, method.hidden_size)
    # zeros, not empty: the output is compared bit-exact, so any row the kernel
    # leaves unwritten must not carry allocator garbage.
    out = torch.zeros(
        x_quant.shape[0], x.shape[-1], dtype=torch.bfloat16, device=x_quant.device
    )
    return trtllm_fp4_block_scale_moe(
        router_logits.to(torch.bfloat16),
        None,
        x_quant,
        x_scale,
        layer.w13_weight,
        layer.w13_weight_scale,
        layer.w13_weight_bias,
        layer.gemm1_alpha,
        layer.gemm1_beta,
        layer.gemm1_clamp_limit,
        layer.w2_weight,
        layer.w2_weight_scale,
        layer.w2_weight_bias,
        None,
        None,
        None,
        layer.num_experts,
        top_k,
        None,
        None,
        method.intermediate_size_per_partition,
        layer.moe_ep_rank * layer.num_local_experts,
        layer.num_local_experts,
        None,
        1,
        True,
        tune_max_num_tokens=next_power_of_2(x_quant.shape[0]),
        output=out,
        enable_pdl=trtllm_moe_enable_pdl(x_quant.shape[0]),
    )[0]


@pytest.mark.parametrize("precision", ["default", "bf16"])
@pytest.mark.parametrize(
    "tokens,num_experts,hidden,inter,top_k",
    [
        (4, 4, 256, 256, 2),
        (16, 8, 512, 512, 2),
        (32, 8, 1024, 1024, 4),
    ],
)
def test_apply_trtllm_gen_matches_flashinfer_direct(
    tokens, num_experts, hidden, inter, top_k, precision, monkeypatch
):
    """``Mxfp4MoEMethod.apply`` (SM100 branch) must produce the same output as a
    direct ``trtllm_fp4_block_scale_moe`` call fed the same inputs.

    Turns red if apply mis-wires a ``FlashInferTrtllmGenMxfp4MoeQuantInfo`` field
    into the kernel (e.g. swapping ``local_expert_offset`` / ``local_num_experts``
    or dropping the bf16-vs-default input-quant branch)."""
    method = _build_method(num_experts, hidden, inter, precision)

    import sglang.srt.layers.moe.moe_runner.flashinfer_trtllm as fi_trtllm_mod

    # Bypass symmetric-memory / TP-group in the fused-func module, where the
    # kernel call now lives.
    monkeypatch.setattr(
        fi_trtllm_mod, "use_symmetric_memory", lambda *a, **kw: nullcontext()
    )
    monkeypatch.setattr(fi_trtllm_mod, "is_allocation_symmetric", lambda: False)
    monkeypatch.setattr(fi_trtllm_mod, "get_tp_group", lambda: None)

    fixtures = _make_random_mxfp4(num_experts, hidden, inter)
    x = torch.randn(tokens, hidden, dtype=torch.bfloat16, device="cuda") * 0.1
    g = torch.Generator(device="cuda").manual_seed(1234)
    router_logits = torch.randn(
        tokens, num_experts, dtype=torch.float32, device="cuda", generator=g
    )
    layer = _build_mock_layer(num_experts, hidden, inter, fixtures)
    layer.moe_runner_config = method.moe_runner_config

    # Convert via the production path so the fixtures can't drift from it.
    method.process_weights_after_loading(layer)

    # ---- FlashInfer-direct reference ----
    out_ref = _ref_trtllm(x, layer, method, precision, top_k, router_logits)

    # ---- SGLang path (same x + router_logits) ----
    out_sglang = method.apply(
        layer, _MockDispatchOutput(x.clone(), router_logits, top_k)
    ).hidden_states

    assert torch.equal(out_sglang, out_ref), (
        f"SGLang vs FlashInfer-direct mismatch (precision={precision}); "
        f"max abs diff = {(out_sglang.float() - out_ref.float()).abs().max().item():.4g}"
    )


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
