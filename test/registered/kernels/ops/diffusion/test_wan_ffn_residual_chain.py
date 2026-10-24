"""Wan blocks hand their FFN residual (MulAdd) to the next block's norm1, which adds it
in the same kernel; a chain of blocks must give the bits of blocks run one by one. The
text K/V projected once per loop must live in the request for that loop only."""

import gc
import sys
import weakref
from contextlib import ExitStack
from unittest.mock import MagicMock, patch

import pytest
import torch
from torch import nn

from sglang.multimodal_gen.configs.sample.sampling_params import SamplingParams
from sglang.multimodal_gen.runtime.distributed.parallel_state import (
    maybe_init_distributed_environment_and_model_parallel,
    model_parallel_is_initialized,
)
from sglang.multimodal_gen.runtime.layers.attention import layer
from sglang.multimodal_gen.runtime.layers.attention.backends.sdpa import SDPABackend
from sglang.multimodal_gen.runtime.managers.forward_context import (
    DENOISING_LOOP_STATE,
    set_forward_context,
)
from sglang.multimodal_gen.runtime.models.dits.wanvideo import (
    _TEXT_KV_STATE,
    WanTransformer3DModel,
    WanTransformerBlock,
)
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.denoising import DenoisingStage
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

    # the cross-attention K/V projected ahead of the step give the same bits
    with ExitStack() as stack:
        for name, value in (
            ("get_attn_backend", SDPABackend),
            ("get_compute_dtype", torch.bfloat16),
            ("_resolve_sp_attention_mode", ("ulysses", False)),
            ("_maybe_install_backend_autotune", None),
        ):
            stack.enter_context(patch.object(layer, name, return_value=value))
        stack.enter_context(set_forward_context(0, None))
        kv = blocks[0].attn2.project_context_kv(context)
        out = blocks[0](x, *args, rope_cos_sin_cache=cache, text_kv=kv)
        ref = blocks[0](x, *args, rope_cos_sin_cache=cache)
    assert torch.equal(out.view(torch.int16), ref.view(torch.int16))


class _TextKVModel(nn.Module):
    _text_kvs = WanTransformer3DModel._text_kvs
    _plain_blocks = WanTransformer3DModel._plain_blocks

    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.blocks = nn.ModuleList(
            WanTransformerBlock(
                dim,
                1024,
                heads,
                qk_norm="rms_norm_across_heads",
                cross_attn_norm=True,
                eps=1e-6,
                supported_attention_backends=set(),
            )
            for _ in range(2)
        )
        with torch.no_grad():
            for p in self.parameters():
                p.normal_(0, 0.02)
        self.layerwise_offload_managers = []


@torch.no_grad()
def test_text_kv_live_for_one_loop_of_one_request():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    if not model_parallel_is_initialized():
        ensure_distributed_env_defaults()
        maybe_init_distributed_environment_and_model_parallel(tp_size=1, sp_size=1)
    dim, heads = 1024, 8
    with ExitStack() as stack:
        for name, value in (
            ("get_attn_backend", SDPABackend),
            ("get_compute_dtype", torch.bfloat16),
            ("_resolve_sp_attention_mode", ("ulysses", False)),
            ("_maybe_install_backend_autotune", None),
        ):
            stack.enter_context(patch.object(layer, name, return_value=value))
        torch.manual_seed(0)
        expert, expert_2 = (
            _TextKVModel(dim, heads).cuda().to(torch.bfloat16) for _ in range(2)
        )
    context = torch.randn(1, 64, dim, device="cuda", dtype=torch.bfloat16)
    positive, negative = torch.randn(2, 1, 64, 4096, device="cuda")
    new_req = lambda: Req(sampling_params=SamplingParams(prompt="test"))

    def text_kvs(model, source, req):
        with set_forward_context(0, None, req):
            return model._text_kvs(source, context)

    def first_k(model, source, req):
        return weakref.ref(text_kvs(model, source, req)[0][0])

    # every step of the loop reads the K/V projected on its first, per CFG branch
    req = new_req()
    kvs, neg_kvs = text_kvs(expert, positive, req), text_kvs(expert, negative, req)
    assert text_kvs(expert, positive, req) is kvs
    assert text_kvs(expert, negative, req) is neg_kvs
    for (k, v), block in zip(kvs, expert.blocks):
        ref_k, ref_v = block.attn2.project_context_kv(context)
        assert torch.equal(k, ref_k) and torch.equal(v, ref_v)

    # another request projects its own; K/V left on another device are projected again
    assert text_kvs(expert, positive, new_req()) is not kvs
    req.extra[DENOISING_LOOP_STATE][_TEXT_KV_STATE][(id(expert), id(positive))] = (
        positive,
        [(k.cpu(), v.cpu()) for k, v in kvs],
    )
    moved = text_kvs(expert, positive, req)
    assert moved[0][0].is_cuda and torch.equal(moved[0][0], kvs[0][0])

    # the second expert's first step frees the first expert's K/V
    gone = [weakref.ref(t) for t in (moved[0][0], neg_kvs[0][0])]
    del kvs, neg_kvs, moved, ref_k, ref_v, k, v
    text_kvs(expert_2, positive, req)
    assert all(ref() is None for ref in gone)

    # the end of the loop frees the rest, before decode
    gone = first_k(expert_2, positive, req)
    stage = MagicMock()
    stage._postprocess_sp_latents.return_value = (None, None)
    stage._sp_world_size.return_value = 1
    DenoisingStage._post_denoising_loop(stage, req, None, [], [], MagicMock(), True)
    assert gone() is None and DENOISING_LOOP_STATE not in req.extra

    # a request that ends mid-loop takes its K/V with it: the model holds none
    req = new_req()
    gone = first_k(expert, positive, req)
    del req
    gc.collect()
    assert gone() is None


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
