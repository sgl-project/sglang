"""Unit test for triton_kernels MXFP4 path in :class:`Mxfp4MoEMethod`.

Builds a single-layer GPT-OSS-style MXFP4 MoE and drives the whole SGLang
plumbing the way ``FusedMoE`` does -- ``create_weights`` -> loader-style copy
into the padded buffers -> ``process_weights_after_loading`` (mxfp4 swizzle)
-> ``apply`` through the ``triton_kernel`` MoE runner -- then compares the
kernel output against a dequantized bf16 reference computed with plain torch.
This test pins the layer, so a broken swizzle / padding / activation wiring
fails in seconds instead of showing up as an eval-score drop for the model.

To run the test:

    python -m pytest test/registered/unit/layers/quantization/test_mxfp4_triton_kernels.py -v
"""

from __future__ import annotations

import sys

import pytest
import torch

from sglang.test.ci.ci_register import register_xpu_ci

register_xpu_ci(est_time=180, suite="stage-b-test-1-gpu-xpu")

pytest.importorskip("triton_kernels")

if not torch.xpu.is_available():
    pytest.skip("XPU required", allow_module_level=True)

from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
from sglang.srt.layers.moe.token_dispatcher.standard import StandardDispatchOutput
from sglang.srt.layers.moe.topk import TritonKernelTopKOutput, routing
from sglang.srt.layers.moe.utils import MoeRunnerBackend
from sglang.srt.layers.quantization.mxfp4 import Mxfp4MoEMethod
from sglang.srt.runtime_context import get_flags
from sglang.srt.utils import get_device

GROUP_SIZE = 32  # MXFP4 block size
# create_weights rounds the expert intermediate up to this before allocating;
# the mxfp4 swizzle in triton_kernels needs the padded width.
PAD_ALIGNMENT = 64
# GPT-OSS SwiGLU scalars (Mxfp4MoEMethod's defaults for this checkpoint).
SWIGLU_ALPHA = 1.702
SWIGLU_LIMIT = 7.0
DEVICE = get_device()


class _MockLayer(torch.nn.Module):
    """Stand-in for ``FusedMoE``: ``create_weights`` registers the expert
    buffers on it and ``apply`` reads the TP/EP ranks off it. Built by hand so
    the test stays out of SGLang's distributed init path (``get_tp_group``
    etc.)."""

    def __init__(self, num_experts: int, hidden: int, inter: int):
        super().__init__()
        self.num_experts = num_experts
        self.num_local_experts = num_experts  # tests run with EP size = 1
        self.hidden_size = hidden
        self.intermediate_size_per_partition = inter
        self.moe_tp_size = 1
        self.moe_tp_rank = 0
        self.moe_ep_size = 1
        self.moe_ep_rank = 0


def _build_method() -> Mxfp4MoEMethod:
    """``Mxfp4MoEMethod.__init__`` resolves its backend flags from the
    published exec config, which a unit test has no live context for; set the
    flags the triton_kernels path reads and nothing else."""
    method = Mxfp4MoEMethod.__new__(Mxfp4MoEMethod)
    method.prefix = "test"
    method.topk_indices_dtype = None
    method.with_bias = True
    method.use_triton_kernels = True
    method.use_flashinfer = False
    method.use_marlin = False
    method.use_deep_gemm = False
    method.use_mega_moe = False
    method.flashinfer_mxfp4_moe_precision = "default"
    method._fi_kernel = None
    return method


def _round_up(x: int, base: int) -> int:
    return ((x + base - 1) // base) * base


def _make_random_mxfp4(num_experts: int, hidden: int, inter: int, seed: int = 0):
    """Checkpoint-shaped MXFP4 tensors: packed e2m1 (2 values per byte, K-major)
    with raw UE8M0 group scales, gate/up rows interleaved as GPT-OSS ships
    them."""
    torch.manual_seed(seed)
    w13 = torch.randint(
        0, 256, (num_experts, 2 * inter, hidden // 2), dtype=torch.uint8, device=DEVICE
    )
    w2 = torch.randint(
        0, 256, (num_experts, hidden, inter // 2), dtype=torch.uint8, device=DEVICE
    )
    # E8M0 scales centered on 127 (= 2^0); a narrow band keeps the dequantized
    # values in a sane range so the SwiGLU clamp doesn't dominate the output.
    w13_s = torch.randint(
        125,
        130,
        (num_experts, 2 * inter, hidden // GROUP_SIZE),
        dtype=torch.uint8,
        device=DEVICE,
    )
    w2_s = torch.randint(
        125,
        130,
        (num_experts, hidden, inter // GROUP_SIZE),
        dtype=torch.uint8,
        device=DEVICE,
    )
    w13_b = (
        torch.randn(num_experts, 2 * inter, dtype=torch.float32, device=DEVICE) * 0.01
    ).to(torch.bfloat16)
    w2_b = (
        torch.randn(num_experts, hidden, dtype=torch.float32, device=DEVICE) * 0.01
    ).to(torch.bfloat16)
    return w13, w2, w13_s, w2_s, w13_b, w2_b


def _load_weights(layer, w13, w2, w13_s, w2_s, w13_b, w2_b):
    """Mirror the mxfp4 weight loader's naive copy: the checkpoint's unpadded
    rows land in the leading slice of the padded buffer and the rest keeps the
    neutral fill create_weights allocated."""
    rows = w13.shape[1]
    layer.w13_weight.data[:, :rows, :] = w13
    layer.w13_weight_scale.data[:, :rows, :] = w13_s
    layer.w13_weight_bias.data[:, :rows] = w13_b
    inter = w2.shape[2] * 2  # packed 4-bit -> *2 for the raw intermediate
    layer.w2_weight.data[:, :, : inter // 2] = w2
    layer.w2_weight_scale.data[:, :, : inter // GROUP_SIZE] = w2_s
    layer.w2_weight_bias.data.copy_(w2_b)


def _create_layer(method, num_experts: int, hidden: int, inter: int):
    layer = _MockLayer(num_experts, hidden, inter)
    # create_weights takes no device argument; it allocates on the ambient one.
    with torch.device(DEVICE):
        method.create_weights(
            layer=layer,
            num_experts=num_experts,
            hidden_size=hidden,
            intermediate_size_per_partition=inter,
            params_dtype=torch.bfloat16,
            with_bias=True,
        )
    return layer


def _runner_config(num_experts: int, hidden: int, inter: int, top_k: int):
    return MoeRunnerConfig(
        num_experts=num_experts,
        num_local_experts=num_experts,
        hidden_size=hidden,
        intermediate_size_per_partition=inter,
        top_k=top_k,
        activation="silu",
        is_gated=True,
        gemm1_alpha=SWIGLU_ALPHA,
        gemm1_clamp_limit=SWIGLU_LIMIT,
    )


def _prepare(num_experts: int, hidden: int, inter: int, top_k: int, seed: int = 0):
    """Run the load-time half of the path and return the method / layer plus
    the checkpoint tensors the reference needs."""
    method = _build_method()
    layer = _create_layer(method, num_experts, hidden, inter)
    checkpoint = _make_random_mxfp4(num_experts, hidden, inter, seed=seed)
    _load_weights(layer, *checkpoint)
    with get_flags().moe.override(runner_backend=MoeRunnerBackend.TRITON_KERNELS):
        method.create_moe_runner(
            layer, _runner_config(num_experts, hidden, inter, top_k)
        )
        method.process_weights_after_loading(layer)
    return method, layer, checkpoint


def _dequant(weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Reference dequantization, via the same helper the bf16-upcast branch of
    ``process_weights_after_loading`` uses."""
    from triton_kernels.numerics_details.mxfp import upcast_from_mxfp

    return upcast_from_mxfp(weight, scale, target_dtype=torch.bfloat16, axis=-1)


def _reference_moe(x, w13_deq, w2_deq, w13_b, w2_b, topk_weights, topk_ids):
    """Dequantized reference for the GPT-OSS MXFP4 expert: interleaved gate/up
    rows, gate clamped from above and up clamped both ways, then the
    routing-weighted sum over the selected experts. The routing weight scales
    the down-proj bias too -- ``matmul``'s epilogue applies ``gammas`` after the
    bias add, matching HF's ``(down(act) + down_bias) * routing_weight``."""
    out = torch.zeros(x.shape, dtype=torch.float32, device=x.device)
    for token in range(x.shape[0]):
        for slot in range(topk_ids.shape[1]):
            expert = int(topk_ids[token, slot])
            gate_up = x[token].float() @ w13_deq[expert].float().t()
            gate_up = gate_up + w13_b[expert].float()
            gate = gate_up[0::2].clamp(max=SWIGLU_LIMIT)
            up = gate_up[1::2].clamp(min=-SWIGLU_LIMIT, max=SWIGLU_LIMIT)
            act = gate * torch.sigmoid(gate * SWIGLU_ALPHA) * (up + 1)
            # The kernel writes the activation out in bf16 before the second
            # GEMM; round here too so only accumulation order is left to differ.
            act = act.to(torch.bfloat16).float()
            expert_out = act @ w2_deq[expert].float().t() + w2_b[expert].float()
            out[token] += topk_weights[token, slot].float() * expert_out
    return out


def _dispatch_output(x, logits, top_k):
    """``routing`` is what ``TopK.forward`` calls for the triton_kernel output
    format; calling it directly keeps the test out of the distributed path."""
    topk_output = TritonKernelTopKOutput(*routing(logits, top_k, sm_first=False))
    return StandardDispatchOutput(x, None, topk_output)


def _reference_topk(logits, top_k):
    # sm_first=False makes triton_kernels softmax the selected logits, which is
    # SGLang's renormalize=True (see the note at the routing() call in topk.py).
    values, ids = torch.topk(logits.float(), top_k, dim=-1)
    return torch.softmax(values, dim=-1), ids


def _relative_error(actual: torch.Tensor, expected: torch.Tensor) -> float:
    return ((actual.float() - expected.float()).norm() / expected.float().norm()).item()


@pytest.mark.parametrize(
    "num_experts,hidden,inter",
    [
        # Already 64-aligned: the buffer must keep the checkpoint's width.
        (4, 256, 256),
        # 160 -> 192: the pad lands past the loaded rows, on both projections.
        (4, 384, 160),
    ],
)
def test_create_weights_pads_intermediate_only(num_experts, hidden, inter):
    """The triton_kernels swizzle needs a 64-aligned expert intermediate, and
    the mxfp4 loader copies the checkpoint in naively -- so the padding has to
    grow the intermediate dim only. Padding the hidden dim instead would shift
    the rows the loader writes and silently mis-place every expert."""
    method = _build_method()
    layer = _create_layer(method, num_experts, hidden, inter)

    padded = _round_up(inter, PAD_ALIGNMENT)
    assert method.intermediate_size_per_partition == padded
    assert method.hidden_size == hidden
    assert layer.w13_weight.shape == (num_experts, 2 * padded, hidden // 2)
    assert layer.w13_weight_scale.shape == (
        num_experts,
        2 * padded,
        hidden // GROUP_SIZE,
    )
    assert layer.w2_weight.shape == (num_experts, hidden, padded // 2)
    assert layer.w2_weight_scale.shape == (
        num_experts,
        hidden,
        padded // GROUP_SIZE,
    )
    # 127 is the neutral UE8M0 exponent; a fresh buffer has to dequantize to a
    # valid MXFP4 tensor because the swizzle runs over the pad rows too.
    assert torch.all(layer.w13_weight_scale == 127)
    assert torch.all(layer.w2_weight_scale == 127)


def test_process_weights_installs_swizzled_tensors():
    """``apply`` reads the swizzled weights off the method, not the layer:
    ``process_weights_after_loading`` moves them to
    ``w13/w2_weight_triton_tensor`` and drops the layer parameters. If a
    refactor stops doing either, ``apply`` either raises or hands the kernel
    the raw packed bytes with no scales attached."""
    num_experts, hidden, inter = 4, 256, 256
    method, layer, _ = _prepare(num_experts, hidden, inter, top_k=2)

    assert method.w13_weight_triton_tensor is not None
    assert method.w2_weight_triton_tensor is not None
    assert not hasattr(layer, "w13_weight")
    assert not hasattr(layer, "w2_weight")
    # triton_kernels' matmul adds the bias in fp32.
    assert layer.w13_weight_bias.dtype == torch.float32
    assert layer.w2_weight_bias.dtype == torch.float32
    for precision_config in (method.w13_precision_config, method.w2_precision_config):
        assert precision_config.b_mx_scale is not None
        # Present only on triton_kernels builds that take the block size
        # explicitly; when it is, the MXFP4 group size is the only valid value.
        if hasattr(precision_config, "b_microblock_size"):
            assert precision_config.b_microblock_size == GROUP_SIZE


@pytest.mark.parametrize(
    "tokens,num_experts,hidden,inter,top_k",
    [
        (16, 4, 256, 256, 2),
        (8, 8, 384, 160, 4),
        (64, 4, 256, 256, 1),
    ],
)
def test_apply_matches_dequantized_reference(tokens, num_experts, hidden, inter, top_k):
    """End-to-end: the swizzled MXFP4 kernel must agree with the same MoE
    computed from dequantized bf16 weights. Covers the swizzle layout, the
    scale layout, the padded rows staying inert, and the GPT-OSS SwiGLU
    scalars reaching the fused activation."""
    method, layer, checkpoint = _prepare(num_experts, hidden, inter, top_k, seed=1)
    w13, w2, w13_s, w2_s, w13_b, w2_b = checkpoint

    torch.manual_seed(2)
    x = torch.randn(tokens, hidden, dtype=torch.bfloat16, device=DEVICE) * 0.1
    logits = torch.randn(tokens, num_experts, dtype=torch.float32, device=DEVICE)

    with get_flags().moe.override(runner_backend=MoeRunnerBackend.TRITON_KERNELS):
        actual = method.apply(layer, _dispatch_output(x, logits, top_k)).hidden_states

    topk_weights, topk_ids = _reference_topk(logits, top_k)
    expected = _reference_moe(
        x,
        _dequant(w13, w13_s),
        _dequant(w2, w2_s),
        w13_b,
        w2_b,
        topk_weights,
        topk_ids,
    )

    error = _relative_error(actual, expected)
    assert error < 2e-2, f"relative L2 error {error:.4g} vs dequantized reference"


def test_apply_rejects_expert_parallel():
    """triton_kernels has no expert-parallel dispatch, and the runner takes the
    expert ids as global ones -- EP > 1 has to fail loudly instead of routing
    tokens to the wrong local expert."""
    method, layer, _ = _prepare(4, 256, 256, top_k=2)
    layer.moe_ep_size = 2

    torch.manual_seed(3)
    x = torch.randn(8, 256, dtype=torch.bfloat16, device=DEVICE) * 0.1
    logits = torch.randn(8, 4, dtype=torch.float32, device=DEVICE)

    with get_flags().moe.override(runner_backend=MoeRunnerBackend.TRITON_KERNELS):
        with pytest.raises(AssertionError, match="Expert parallel is not supported"):
            method.apply(layer, _dispatch_output(x, logits, 2))


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
