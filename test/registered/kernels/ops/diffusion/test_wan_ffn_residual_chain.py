"""Wan blocks hand their FFN residual (MulAdd) to the next block's norm1, which adds it
in the same kernel; a chain of blocks must give the bits of blocks run one by one."""

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
def test_chained_blocks_match_blocks_one_by_one():
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
        blocks = []
        for _ in range(3):
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
            blocks.append(block.to(torch.bfloat16))
        assert all(block.chains_ffn_residual for block in blocks)
        x = torch.randn(1, seq, dim, device="cuda", dtype=torch.bfloat16) * 4
        context = torch.randn(1, 64, dim, device="cuda", dtype=torch.bfloat16)
        temb = torch.randn(1, 6, dim, device="cuda")
        theta = torch.rand(seq, 64, device="cuda") * 50
        freqs = (theta.cos(), theta.sin())
        cache = torch.cat(freqs, dim=-1).contiguous()
        args = (context, temb, freqs)

        ref = x
        for block in blocks:
            ref = block(ref, *args, rope_cos_sin_cache=cache)

        hidden, residual = x, None
        for block in blocks[:-1]:
            hidden, residual = block(
                hidden,
                *args,
                rope_cos_sin_cache=cache,
                ffn_residual=residual,
                defer_ffn_residual=True,
            )
        out = blocks[-1](hidden, *args, rope_cos_sin_cache=cache, ffn_residual=residual)
    assert torch.equal(out.view(torch.int16), ref.view(torch.int16))


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
