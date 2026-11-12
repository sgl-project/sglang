"""Unified MXFP8 payload, scales, page moves, and FA4 attention parity."""

from types import SimpleNamespace

import pytest
import torch

from sglang.srt.mem_cache.memory_pool import KVWriteLoc
from sglang.srt.utils import get_device_sm
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=420, stage="base-b-kernel-unit", runner_config="4-gpu-b200")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or get_device_sm() < 100, reason="Requires Blackwell"
)
PS, HD = 128, 128


def _make_unified_pool(heads):
    from sglang.srt.mem_cache.kv.layout.mha import MHASubPoolSpec, MHAView
    from sglang.test.unified_allocator_fixtures import arena_of

    specs = [
        MHASubPoolSpec(
            name=name,
            layer_num=2,
            head_num=heads,
            head_dim=HD,
            store_dtype=torch.uint8,
            kv_cache_dtype=torch.float8_e4m3fn,
            is_mxfp8=True,
            grow_direction=grow,
        )
        for name, grow in (("full", "down"), ("swa", "up"))
    ]
    arena = arena_of(
        specs, total_bytes=8 * PS * specs[0].entry_bytes(), device="cuda", page_size=PS
    )
    return MHAView(arena=arena, spec=specs[0], page_size=PS, model_layer_ids=(0, 1))


@pytest.mark.parametrize("pack_gqa", [False, True], ids=["unpacked", "packed"])
@pytest.mark.parametrize(
    "relative_bias,bias_dtype",
    [(False, torch.float32), (True, torch.float32), (True, torch.bfloat16)],
    ids=["unbiased", "fp32_bias", "bf16_bias"],
)
@pytest.mark.parametrize("query_tokens,num_splits", [(1, 1), (8, 1), (129, 1), (8, 32)])
@pytest.mark.parametrize("window", [-1, 127], ids=["full", "swa128"])
def test_fa4_matches_mxfp8_and_bf16_references(
    query_tokens, num_splits, window, relative_bias, bias_dtype, pack_gqa
):
    from sglang.kernels.ops.attention.flash_attention_v4 import flash_attn_with_kvcache
    from sglang.kernels.ops.quantization.mxfp8_quant import to_mxfp8
    from sglang.srt.mem_cache.layout.paged_view import paged_view
    from sglang.srt.mem_cache.memory_pool import MHATokenToKVPoolMXFP8

    torch.manual_seed(7)
    heads = 2
    group = 8 if pack_gqa else 6
    query_heads = heads * group
    unified = _make_unified_pool(heads)
    static = MHATokenToKVPoolMXFP8(
        size=unified.size,
        page_size=PS,
        dtype=torch.float8_e4m3fn,
        head_num=heads,
        head_dim=HD,
        layer_num=1,
        device="cuda",
        enable_memory_saver=False,
    )
    loc = torch.arange(PS, 4 * PS, device="cuda")
    k, v = [
        torch.randn(loc.numel(), heads, HD, device="cuda", dtype=torch.bfloat16)
        for _ in range(2)
    ]
    for pool in (unified, static):
        pool.set_kv_buffer(
            SimpleNamespace(layer_id=0), KVWriteLoc(loc, physical=True), k, v
        )
    q = to_mxfp8(
        torch.randn(query_tokens, query_heads, HD, device="cuda", dtype=torch.bfloat16)
    )
    kwargs = dict(
        q=q.data,
        sfq=q.scale.view(torch.float8_e8m0fnu),
        page_table=torch.tensor([[1, 2, 3]], dtype=torch.int32, device="cuda"),
        cache_seqlens=torch.tensor([loc.numel()], dtype=torch.int32, device="cuda"),
        cu_seqlens_q=torch.tensor([0, query_tokens], dtype=torch.int32, device="cuda"),
        max_seqlen_q=query_tokens,
        causal=True,
        window_size=(window, 0),
        num_splits=num_splits,
        pack_gqa=pack_gqa,
    )
    if relative_bias:
        from sglang.srt.models.inkling_common.attn import (
            get_inkling_relative_attention_score_mod,
        )

        kwargs.update(
            score_mod=get_inkling_relative_attention_score_mod(1024),
            aux_tensors=[
                torch.randn(
                    query_tokens, query_heads, 1024, device="cuda", dtype=bias_dtype
                )
            ],
        )
    outputs = []
    for pool in (unified, static):
        sk, sv = pool.get_kv_scale_buffer(0)
        outputs.append(
            flash_attn_with_kvcache(
                k_cache=paged_view(pool.get_key_buffer(0), PS),
                v_cache=paged_view(pool.get_value_buffer(0), PS),
                sfk=sk,
                sfv=sv,
                out=torch.empty_like(q.data, dtype=torch.float32),
                **kwargs,
            )
        )
    torch.testing.assert_close(outputs[0], outputs[1], atol=3e-3, rtol=3e-3)

    def dequantize(data, scales):
        scale = torch.exp2(scales.view(torch.uint8).float() - 127)
        return (
            data.view(torch.float8_e4m3fn).float() * scale.repeat_interleave(32, dim=-1)
        ).bfloat16()

    reference_kwargs = {
        key: value for key, value in kwargs.items() if key not in ("q", "sfq")
    }
    reference = flash_attn_with_kvcache(
        q=dequantize(q.data, q.scale),
        k_cache=paged_view(
            dequantize(unified.k_buffer[0], unified.k_scale_buffer[0]), PS
        ),
        v_cache=paged_view(
            dequantize(unified.v_buffer[0], unified.v_scale_buffer[0]), PS
        ),
        out=torch.empty_like(q.data, dtype=torch.float32),
        **reference_kwargs,
    )
    torch.testing.assert_close(outputs[0], reference, atol=3e-3, rtol=3e-3)

    if relative_bias and query_tokens == 1 and window == -1:
        # Verify the relative-bias convention independently of FA4.
        query = dequantize(q.data, q.scale).float()
        keys = dequantize(unified.k_buffer[0], unified.k_scale_buffer[0])[loc]
        values = dequantize(unified.v_buffer[0], unified.v_scale_buffer[0])[loc]
        keys = keys.repeat_interleave(group, dim=1).float()
        values = values.repeat_interleave(group, dim=1).float()
        scores = torch.einsum("thd,shd->hts", query, keys) / HD**0.5
        bias = kwargs["aux_tensors"][0][..., : loc.numel()].flip(-1).transpose(0, 1)
        expected = torch.einsum("hts,shd->thd", (scores + bias).softmax(-1), values)
        torch.testing.assert_close(outputs[0], expected, atol=3e-3, rtol=3e-3)
