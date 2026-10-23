"""Wan's q/k/v projections share one weight after loading and run as one GEMM;
the block's output must keep its bits, and any layer that stops reading the
shared weight must send the block back to its own three projections."""

import sys
from contextlib import ExitStack
from unittest.mock import patch

import pytest
import torch

from sglang.multimodal_gen.runtime.distributed.parallel_state import (
    maybe_init_distributed_environment_and_model_parallel,
    model_parallel_is_initialized,
)
from sglang.multimodal_gen.runtime.layers.attention import layer
from sglang.multimodal_gen.runtime.layers.attention.backends.sdpa import SDPABackend
from sglang.multimodal_gen.runtime.managers.forward_context import set_forward_context
from sglang.multimodal_gen.runtime.models.dits.wanvideo import WanTransformerBlock
from sglang.multimodal_gen.test.single_test_file.component_accuracy.utils import (
    ensure_distributed_env_defaults,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="1-gpu-large")


@torch.no_grad()
def test_fused_qkv_keeps_bits_and_falls_back():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
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
        block = block.to(torch.bfloat16)
        x = torch.randn(1, seq, dim, device="cuda", dtype=torch.bfloat16)
        context = torch.randn(1, 64, dim, device="cuda", dtype=torch.bfloat16)
        temb = torch.randn(1, 6, dim, device="cuda")
        theta = torch.rand(seq, 64, device="cuda") * 50
        cos, sin = theta.cos(), theta.sin()
        cache = torch.cat([cos, sin], dim=-1).contiguous()
        run = lambda: block(x, context, temb, (cos, sin), rope_cos_sin_cache=cache)

        weights = [proj.weight.clone() for proj in (block.to_q, block.to_k, block.to_v)]
        ref = run()
        assert not block._qkv_fused(x)
        block.fuse_qkv_weights()
        assert block._qkv_fused(x)
        for proj, w in zip((block.to_q, block.to_k, block.to_v), weights):
            assert torch.equal(proj.weight, w)
        fused = run()
        assert torch.equal(fused.view(torch.int16), ref.view(torch.int16))

        # a layer whose weight moved (offload, a weight swap) stops the fused GEMM
        block.to_k.weight.data = block.to_k.weight.data.clone()
        assert not block._qkv_fused(x)
        assert torch.equal(run().view(torch.int16), ref.view(torch.int16))


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
