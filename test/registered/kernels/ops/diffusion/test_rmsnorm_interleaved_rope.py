"""``fused_rmsnorm_interleaved_rope_``: Wan's across-heads q/k RMSNorm and its
interleaved RoPE in one pass, held bit-exact to the two kernels it replaces
(``flashinfer.norm.rmsnorm`` then FlashInfer's interleaved cos/sin-cache RoPE)."""

import sys
from contextlib import ExitStack
from unittest.mock import patch

import pytest
import torch
from flashinfer.norm import rmsnorm

from sglang.kernels.ops.diffusion import (
    can_use_fused_rmsnorm_interleaved_rope,
    fused_rmsnorm_interleaved_rope_,
)
from sglang.multimodal_gen.runtime.distributed.parallel_state import (
    maybe_init_distributed_environment_and_model_parallel,
    model_parallel_is_initialized,
)
from sglang.multimodal_gen.runtime.layers.attention import layer
from sglang.multimodal_gen.runtime.layers.attention.backends.sdpa import SDPABackend
from sglang.multimodal_gen.runtime.layers.rotary_embedding.utils import (
    apply_flashinfer_rope_qk_inplace,
)
from sglang.multimodal_gen.runtime.managers.forward_context import set_forward_context
from sglang.multimodal_gen.runtime.models.dits.wanvideo import WanTransformerBlock
from sglang.multimodal_gen.test.single_test_file.component_accuracy.utils import (
    ensure_distributed_env_defaults,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="1-gpu-large")


def _require_sm90() -> None:
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 9:
        pytest.skip("CuTe-DSL RMSNorm needs sm_90+")


@pytest.mark.parametrize(
    "batch,seq,hidden,head_dim,row_pad",
    [
        (1, 333, 5120, 128, 0),
        (2, 150, 5120, 128, 0),
        (1, 257, 3072, 128, 0),
        (1, 64, 5120, 128, 1024),
    ],
)
def test_matches_rmsnorm_then_rope_bitwise(batch, seq, hidden, head_dim, row_pad):
    _require_sm90()
    g = torch.Generator(device="cuda").manual_seed(seq * hidden + batch)
    eps = 1e-6
    x = (torch.randn(batch * seq, hidden, generator=g, device="cuda") * 3).bfloat16()
    weight = (1 + 0.1 * torch.randn(hidden, generator=g, device="cuda")).bfloat16()
    theta = torch.rand(seq, head_dim // 2, generator=g, device="cuda") * 100
    cache = torch.cat([theta.cos(), theta.sin()], dim=-1).contiguous()

    normed = rmsnorm(x, weight, eps).view(batch, seq, hidden // head_dim, head_dim)
    ref, _ = apply_flashinfer_rope_qk_inplace(
        normed, normed.clone(), cache, is_neox=False
    )

    # rows padded apart, as a slice of a wider projection output
    buf = torch.full(
        (batch * seq, hidden + row_pad), float("nan"), device="cuda"
    ).bfloat16()
    out = buf[:, :hidden]
    out.copy_(x)
    assert can_use_fused_rmsnorm_interleaved_rope(out, weight, cache, head_dim)
    fused_rmsnorm_interleaved_rope_(out, weight, cache, eps, head_dim)
    assert torch.equal(
        out.view(torch.int16), ref.reshape(batch * seq, hidden).view(torch.int16)
    )
    if row_pad:
        assert buf[:, hidden:].isnan().all()


def test_runs_on_the_current_stream():
    """Launched on torch's current stream: its input may still be in flight there."""
    _require_sm90()
    g = torch.Generator(device="cuda").manual_seed(1)
    eps, hidden, head_dim, rows = 1e-6, 5120, 128, 4096
    a = torch.randn(rows, 1024, generator=g, device="cuda").bfloat16()
    b = torch.randn(1024, hidden, generator=g, device="cuda").bfloat16()
    weight = (1 + 0.1 * torch.randn(hidden, generator=g, device="cuda")).bfloat16()
    theta = torch.rand(rows, head_dim // 2, generator=g, device="cuda")
    cache = torch.cat([theta.cos(), theta.sin()], dim=-1).contiguous()
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(3):
            x = a @ b
            ref = rmsnorm(x, weight, eps).view(1, rows, hidden // head_dim, head_dim)
            ref, _ = apply_flashinfer_rope_qk_inplace(
                ref, ref.clone(), cache, is_neox=False
            )
            out = a @ b
            fused_rmsnorm_interleaved_rope_(out, weight, cache, eps, head_dim)
            assert torch.equal(
                out.view(torch.int16), ref.reshape(rows, hidden).view(torch.int16)
            )


@torch.no_grad()
def test_wan_block_fused_qk_norm_rope_is_bit_exact():
    """A Wan block gives the same bits with the q/k norm and RoPE fused or not."""
    _require_sm90()
    if not model_parallel_is_initialized():
        ensure_distributed_env_defaults()
        maybe_init_distributed_environment_and_model_parallel(tp_size=1, sp_size=1)
    seq, dim, heads = 512, 5120, 40
    with ExitStack() as stack:
        for name, value in (
            ("get_attn_backend", SDPABackend),
            ("get_compute_dtype", torch.bfloat16),
            ("_resolve_sp_attention_mode", ("ulysses", False)),
            ("_maybe_install_backend_autotune", None),
        ):
            stack.enter_context(patch.object(layer, name, return_value=value))
        stack.enter_context(set_forward_context(0, None))
        torch.manual_seed(0)
        block = WanTransformerBlock(
            dim,
            1024,
            heads,
            qk_norm="rms_norm_across_heads",
            cross_attn_norm=True,
            eps=1e-6,
            supported_attention_backends=set(),
        ).cuda()
        for p in block.parameters():
            p.normal_(0, 0.02)
        block.norm_q.weight.normal_(1, 0.1)
        block.norm_k.weight.normal_(1, 0.1)
        block = block.to(torch.bfloat16)
        assert block.fuse_qk_norm_rope
        x = torch.randn(1, seq, dim, device="cuda", dtype=torch.bfloat16)
        context = torch.randn(1, 64, dim, device="cuda", dtype=torch.bfloat16)
        temb = torch.randn(1, 6, dim, device="cuda")
        theta = torch.rand(seq, 64, device="cuda") * 50
        cos, sin = theta.cos(), theta.sin()
        cache = torch.cat([cos, sin], dim=-1).contiguous()
        outs = []
        for fuse in (False, True):
            block.fuse_qk_norm_rope = fuse
            outs.append(block(x, context, temb, (cos, sin), rope_cos_sin_cache=cache))
    assert torch.equal(outs[0].view(torch.int16), outs[1].view(torch.int16))


def test_rejects_unsupported_inputs():
    _require_sm90()
    x = torch.zeros(8, 5120, device="cuda", dtype=torch.bfloat16)
    w = torch.ones(5120, device="cuda", dtype=torch.bfloat16)
    cache = torch.zeros(8, 128, device="cuda")
    assert can_use_fused_rmsnorm_interleaved_rope(x, w, cache, 128)
    assert not can_use_fused_rmsnorm_interleaved_rope(x.float(), w.float(), cache, 128)
    assert not can_use_fused_rmsnorm_interleaved_rope(x, w.float(), cache, 128)
    assert not can_use_fused_rmsnorm_interleaved_rope(x, w, cache.bfloat16(), 128)
    assert not can_use_fused_rmsnorm_interleaved_rope(x, w, cache[:3], 128)
    assert not can_use_fused_rmsnorm_interleaved_rope(
        x[:, 1:5113], w[:5112], cache, 128
    )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
