"""MHA prefix restoration must project cached and newly extended KV rows."""

import sys
from functools import partial
from types import SimpleNamespace
from unittest import mock

import pytest
import torch
import torch.nn.functional as F

from sglang.srt.models.deepseek_common.attention_forward_methods import forward_mha_rocm
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu", nightly=False, disabled=None)


@pytest.mark.parametrize("prefix_lens", [(0, 0), (2, 0), (2, 1)])
@pytest.mark.parametrize("cache_path", ["bf16", "fp8", "dcp"])
@pytest.mark.parametrize("weight_layout", ["block_scale", "serialized_fp8", "bf16"])
def test_mha_projects_full_prefix_kv(prefix_lens, cache_path, weight_layout):
    # Two requests, with distinct cached prefixes and newly appended suffixes.
    suffix_lens = (2, 3)
    seq_lens = [p + e for p, e in zip(prefix_lens, suffix_lens)]
    total = sum(seq_lens)
    rank = 256  # Two 128-element quantization blocks.
    latent = torch.arange(1, total * (rank + 1) + 1, dtype=torch.float32).view(
        total, rank + 1
    )
    norm_weight = torch.linspace(0.5, 1.5, rank)
    normalized = F.rms_norm(latent[:, :rank], (rank,), norm_weight, 1e-5)
    projection = torch.arange(1, rank * 6 + 1, dtype=torch.float32).view(rank, 6)
    indices = []
    offset = 0
    for prefix, length in zip(prefix_lens, seq_lens):
        indices.extend(range(offset + prefix, offset + length))
        offset += length
    new_indices = torch.tensor(indices)
    new_latent = latent[new_indices].clone()
    # Cached prefix values are already normalized; suffix entries are filled by
    # the real prepare method's cache-write boundary before the cache is read.
    cached_kv = normalized.clone()
    cached_rope = latent[:, rank:].unsqueeze(1).clone()
    cached_kv[new_indices] = float("nan")
    cached_rope[new_indices] = float("nan")
    projected_inputs = []

    def project(value):
        projected_inputs.append(value)
        unquantized = value[0] if isinstance(value, tuple) else value
        return unquantized @ projection, None

    kv_b_proj = project
    kv_b_proj.weight = torch.empty(
        6,
        rank,
        dtype=torch.bfloat16 if weight_layout == "bf16" else torch.float8_e4m3fn,
    )
    if weight_layout == "block_scale":
        kv_b_proj.weight_scale = torch.ones(1, rank // 128)
    elif weight_layout == "serialized_fp8":
        # Standard serialized FP8 exposes weight_scale_inv. Preserve its normal
        # input-quantization path, which does not use the fused RMSNorm tuple.
        kv_b_proj.weight_scale_inv = torch.ones(1, rank // 128)
    fused_layout = weight_layout == "block_scale"
    norm = mock.Mock(side_effect=lambda x: F.rms_norm(x, (rank,), norm_weight, 1e-5))
    norm.weight = norm_weight
    norm.variance_epsilon = 1e-5

    def fused_norm_quant(value, weight, eps, *args, **kwargs):
        result = F.rms_norm(value, (rank,), weight, eps)
        # Model the quantization boundary without requiring an AMD device.
        return (result, torch.ones(len(result), 1)), result, None, None

    def write_cache(latent_cache, kv_a, k_pe, batch):
        cached_kv[new_indices] = kv_a
        cached_rope[new_indices] = k_pe

    read_cache = mock.Mock(side_effect=lambda *args: (cached_kv, cached_rope))
    layer = SimpleNamespace(
        q_lora_rank=None,
        kv_lora_rank=rank,
        num_local_heads=2,
        qk_head_dim=3,
        qk_nope_head_dim=2,
        qk_rope_head_dim=1,
        v_head_dim=1,
        q_proj=lambda x: (torch.ones(len(x), 6), None),
        kv_a_proj_with_mqa=lambda x: (new_latent, None),
        kv_a_layernorm=norm,
        kv_b_proj=kv_b_proj,
        rotary_emb=None,
        use_dsa=True,
        kv_cache_dtype="fp8_e4m3" if cache_path == "fp8" else "bfloat16",
        current_attention_backend="dsa",
        attn_mha=object(),
        _set_mla_kv_buffer_rocm=write_cache,
        _get_mla_kv_buffer_rocm=read_cache,
        _get_mla_kv_buffer_from_fp8_for_dsa=read_cache,
    )
    mixin = forward_mha_rocm.DeepseekMHARocmForwardMixin
    layer._concat_and_cast_mha_k_rocm = partial(
        mixin._concat_and_cast_mha_k_rocm, layer
    )
    batch = SimpleNamespace(
        mha_one_shot=True,
        extend_prefix_lens_cpu=prefix_lens,
        extend_prefix_lens=torch.tensor(prefix_lens),
        extend_seq_lens=torch.tensor(suffix_lens),
        seq_lens=torch.tensor(seq_lens),
        attn_dcp_metadata=SimpleNamespace(dcp_local_prefix_kv_indices=None),
        fetch_mha_one_shot_kv_indices=lambda: torch.arange(total),
    )
    with (
        mock.patch.object(forward_mha_rocm, "_use_aiter_gfx95", True),
        mock.patch.object(forward_mha_rocm, "_use_aiter_bpreshuffle_gfx95", False),
        mock.patch.object(forward_mha_rocm, "_use_fp8_prefill_attn", False),
        mock.patch.object(
            forward_mha_rocm,
            "fused_rms_fp8_group_quant",
            side_effect=fused_norm_quant,
            create=True,
        ) as fused,
        mock.patch.object(
            forward_mha_rocm, "resolve_attn_backend", return_value=object()
        ),
        mock.patch.object(
            forward_mha_rocm, "get_token_to_kv_pool", return_value=object()
        ),
        mock.patch.object(
            forward_mha_rocm,
            "get_parallel",
            return_value=SimpleNamespace(dcp_enabled=cache_path == "dcp"),
        ),
        mock.patch.object(
            forward_mha_rocm,
            "get_exec",
            return_value=SimpleNamespace(
                kernel=SimpleNamespace(
                    dsa_decode_backend="triton", dsa_prefill_backend="fa3"
                )
            ),
        ),
        mock.patch.object(
            forward_mha_rocm,
            "all_gather_kv_cache_for_mha_extend",
            side_effect=read_cache,
        ),
    ):
        q, k, v, returned_batch = mixin.forward_normal_rocm_prepare(
            layer, torch.arange(len(indices)), torch.empty(len(indices), 1), batch, None
        )

    # A full-prefill reference verifies row order, cached prefix values and the
    # absence of a second RMSNorm on restored KV, not just the projection shape.
    expected = (normalized @ projection).view(total, 2, 3)
    expected_k = torch.cat(
        (expected[..., :2], latent[:, None, rank:].expand(-1, 2, -1)), -1
    )
    torch.testing.assert_close(k, expected_k)
    torch.testing.assert_close(v, expected[..., 2:])
    assert q.shape == (sum(suffix_lens), 2, 3)
    assert returned_batch is batch
    assert len(projected_inputs) == 1
    assert isinstance(projected_inputs[0], tuple) == (
        fused_layout and not any(prefix_lens)
    )
    assert read_cache.call_count == int(any(prefix_lens))
    assert fused.call_count == int(fused_layout)
    assert norm.call_count == int(not fused_layout)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
