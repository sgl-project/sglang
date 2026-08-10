# SPDX-License-Identifier: Apache-2.0
"""Mixed-precision weight and TP/Ulysses numerical contracts for H3 DiT."""

from unittest.mock import patch

import pytest
import torch

from sglang.multimodal_gen.configs.models.dits.minimax_h3 import (
    MiniMaxH3DiTArchConfig,
    MiniMaxH3DiTConfig,
)
from sglang.multimodal_gen.runtime.distributed.parallel_state import (
    maybe_init_distributed_environment_and_model_parallel,
    model_parallel_is_initialized,
)
from sglang.multimodal_gen.runtime.layers.attention.backends.sdpa import SDPAImpl
from sglang.multimodal_gen.runtime.layers.linear import UnquantizedLinearMethod
from sglang.multimodal_gen.runtime.layers.quantization.fp8 import (
    Fp8Config,
    Fp8LinearMethod,
)
from sglang.multimodal_gen.runtime.layers.usp import _usp_input_all_to_all_packed_qkv
from sglang.multimodal_gen.runtime.loader.utils import get_param_names_mapping
from sglang.multimodal_gen.runtime.models.dits.minimax_h3 import (
    MINIMAX_H3_FP32_BUFFER_NAMES,
    MINIMAX_H3_FP32_PARAM_NAMES,
    MiniMaxH3DiTModel,
    _copy_grouped_qkv_tp_shard,
    _reorder_grouped_qkv_to_qkv,
)
from sglang.multimodal_gen.test.single_test_file.component_accuracy.utils import (
    ensure_distributed_env_defaults,
)


def _ensure_single_process_parallel_runtime() -> None:
    if model_parallel_is_initialized():
        return
    ensure_distributed_env_defaults()
    maybe_init_distributed_environment_and_model_parallel(tp_size=1, sp_size=1)


def test_native_weight_names_and_grouped_qkv_reorder():
    arch = MiniMaxH3DiTArchConfig()
    assert arch.param_names_mapping == {}
    assert arch.reverse_param_names_mapping == {}
    mapping = get_param_names_mapping(arch.param_names_mapping)
    for key in (
        "video_patch_proj.weight",
        "token_refiner.blocks.0.attn.qkv_proj.weight",
        "blocks.49.attn.qkv_proj.weight",
        "final_layer.audio_out.weight",
    ):
        assert mapping(key) == (key, None, None)

    weight = torch.arange(12, dtype=torch.float32).reshape(12, 1)
    actual = _reorder_grouped_qkv_to_qkv(
        weight,
        num_query_groups=2,
        heads_per_group=1,
        head_dim=2,
    )
    expected = torch.tensor(
        [0, 1, 6, 7, 2, 3, 8, 9, 4, 5, 10, 11],
        dtype=torch.float32,
    ).reshape(12, 1)
    torch.testing.assert_close(actual, expected)

    grouped = torch.arange(48, dtype=torch.int16).reshape(24, 2).view(torch.bfloat16)
    reordered = _reorder_grouped_qkv_to_qkv(
        grouped,
        num_query_groups=4,
        heads_per_group=1,
        head_dim=2,
    )
    for tp_size in (1, 2, 4):
        local_rows = 8 // tp_size
        for tp_rank in range(tp_size):
            start = tp_rank * local_rows
            param = torch.nn.Parameter(
                torch.empty(3 * local_rows, 2, dtype=torch.bfloat16),
                requires_grad=False,
            )
            param.output_dim = 0
            assert _copy_grouped_qkv_tp_shard(
                param,
                grouped,
                num_query_groups=4,
                head_dim=2,
                tp_rank=tp_rank,
                tp_size=tp_size,
            )
            expected_shard = reordered.view(3, 8, 2)[
                :, start : start + local_rows
            ].reshape(-1, 2)
            assert torch.equal(
                param.view(torch.int16), expected_shard.view(torch.int16)
            )


def test_tp_and_ulysses_admission_uses_tp_local_shapes():
    arch = MiniMaxH3DiTArchConfig()
    model = MiniMaxH3DiTModel.__new__(MiniMaxH3DiTModel)
    model._validate_tp_config(arch=arch, tp_size=2)
    model._validate_tp_config(arch=arch, tp_size=8)
    MiniMaxH3DiTModel._validate_sequence_parallel_config(
        arch=arch,
        tp_size=2,
        ulysses_size=4,
        ring_size=1,
    )

    with pytest.raises(ValueError):
        model._validate_tp_config(arch=arch, tp_size=3)
    with pytest.raises(ValueError):
        MiniMaxH3DiTModel._validate_sequence_parallel_config(
            arch=arch,
            tp_size=4,
            ulysses_size=4,
            ring_size=1,
        )
    # ring is implemented now: it splits rows, not heads, so it carries no
    # head-divisibility constraint of its own
    MiniMaxH3DiTModel._validate_sequence_parallel_config(
        arch=arch,
        tp_size=1,
        ulysses_size=1,
        ring_size=2,
    )
    MiniMaxH3DiTModel._validate_sequence_parallel_config(
        arch=arch,
        tp_size=1,
        ulysses_size=8,
        ring_size=2,
    )
    # what ring does constrain is the packed-sequence alignment, which has to
    # divide by the *combined* degree because ring adds an outer row split on
    # top of Ulysses's inner one
    with pytest.raises(ValueError):
        MiniMaxH3DiTModel._validate_sequence_parallel_config(
            arch=arch,
            tp_size=1,
            ulysses_size=8,
            ring_size=3,
        )
    with pytest.raises(ValueError):
        MiniMaxH3DiTModel._validate_sequence_parallel_config(
            arch=arch,
            tp_size=1,
            ulysses_size=1,
            ring_size=0,
        )


def test_meta_model_enforces_mixed_precision_weight_contract():
    expected_fp32 = set(MINIMAX_H3_FP32_PARAM_NAMES) | set(MINIMAX_H3_FP32_BUFFER_NAMES)
    _ensure_single_process_parallel_runtime()
    with torch.device("meta"):
        model = MiniMaxH3DiTModel(
            config=MiniMaxH3DiTConfig(),
            hf_config={},
            quant_config=None,
        )

    assert model._fsdp_mixed_dtype_params
    for name, tensor in model.state_dict().items():
        if name in expected_fp32:
            assert tensor.dtype == torch.float32, name
        elif tensor.is_floating_point():
            assert tensor.dtype == torch.bfloat16, name


def test_online_fp8_keeps_fp32_boundaries_and_ignored_layers_unquantized():
    _ensure_single_process_parallel_runtime()
    with torch.device("meta"):
        model = MiniMaxH3DiTModel(
            config=MiniMaxH3DiTConfig(),
            hf_config={},
            quant_config=Fp8Config(ignored_layers=["blocks.0.attn.out_proj"]),
        )

    assert isinstance(model.blocks[0].attn.qkv_proj.quant_method, Fp8LinearMethod)
    assert isinstance(
        model.blocks[0].attn.out_proj.quant_method, UnquantizedLinearMethod
    )
    for layer in (
        model.video_patch_proj,
        model.audio_patch_proj,
        model.time_embedder.proj_in,
        model.time_embedder.proj_out,
        model.final_layer.video_out,
        model.final_layer.audio_out,
    ):
        assert isinstance(layer.quant_method, UnquantizedLinearMethod)


def test_sdpa_varlen_fallback_matches_naive_packed_reference():
    torch.manual_seed(0)
    heads, dim = 2, 8
    bounds = (0, 5, 6, 13)
    cu = torch.tensor(bounds, dtype=torch.int32)
    q = torch.randn(bounds[-1], heads, dim)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    scale = dim**-0.5

    attention = SDPAImpl(
        num_heads=heads,
        head_size=dim,
        causal=False,
        softmax_scale=scale,
    )
    out = attention.forward_varlen(
        q,
        k,
        v,
        cu_seqlens=cu,
        cu_seqlens_host=bounds,
        max_seqlen=7,
    )
    for start, stop in zip(bounds[:-1], bounds[1:], strict=True):
        seg_q = q[start:stop].transpose(0, 1)
        seg_k = k[start:stop].transpose(0, 1)
        seg_v = v[start:stop].transpose(0, 1)
        expected = (
            torch.softmax(seg_q @ seg_k.transpose(-1, -2) * scale, dim=-1) @ seg_v
        ).transpose(0, 1)
        torch.testing.assert_close(out[start:stop], expected, atol=1e-6, rtol=1e-6)


@patch(
    "sglang.multimodal_gen.runtime.layers.usp.get_ulysses_parallel_world_size",
    return_value=2,
)
def test_packed_qkv_exchange_preserves_rank_and_head_order(_):
    seq, heads, dim = 3, 4, 2
    q_ranks = [
        torch.arange(seq * heads * dim).reshape(seq, heads, dim) + rank * 1000
        for rank in range(2)
    ]
    k_ranks = [q + 100 for q in q_ranks]
    v_ranks = [q + 200 for q in q_ranks]
    target_rank = 1

    def packet(q, k, v, destination):
        head_slice = slice(destination * 2, (destination + 1) * 2)
        return torch.cat((q[:, head_slice], k[:, head_slice], v[:, head_slice]), dim=-1)

    def fake_all_to_all(actual, role=None):
        expected_input = torch.stack(
            [
                packet(q_ranks[0], k_ranks[0], v_ranks[0], destination)
                for destination in range(2)
            ]
        )
        torch.testing.assert_close(actual, expected_input)
        return torch.stack(
            [
                packet(q, k, v, target_rank)
                for q, k, v in zip(q_ranks, k_ranks, v_ranks, strict=True)
            ]
        )

    with patch(
        "sglang.multimodal_gen.runtime.layers.usp._usp_all_to_all_single",
        side_effect=fake_all_to_all,
    ):
        actual = _usp_input_all_to_all_packed_qkv(q_ranks[0], k_ranks[0], v_ranks[0])

    for index, tensors in enumerate((q_ranks, k_ranks, v_ranks)):
        expected = torch.cat(
            [tensor[:, target_rank * 2 : (target_rank + 1) * 2] for tensor in tensors]
        )
        torch.testing.assert_close(actual[index], expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_cuda_ulysses_qkv_pack_is_bit_exact():
    from sglang.kernels.ops.diffusion.triton.ulysses_qkv import (
        pack_qkv_destination_major,
    )

    torch.manual_seed(23)
    rows, world_size, heads, head_size = 65, 8, 56, 128
    qkv = torch.randn(
        rows,
        3 * heads * head_size,
        device="cuda",
        dtype=torch.bfloat16,
    )
    q, k, v = (
        tensor.reshape(rows, heads, head_size)
        for tensor in qkv.split(heads * head_size, dim=-1)
    )
    local_heads = heads // world_size
    expected = torch.empty(
        world_size,
        rows,
        local_heads,
        3 * head_size,
        device="cuda",
        dtype=torch.bfloat16,
    )
    for index, tensor in enumerate((q, k, v)):
        shards = tensor.view(rows, world_size, local_heads, head_size).permute(
            1, 0, 2, 3
        )
        expected[..., index * head_size : (index + 1) * head_size].copy_(shards)

    actual = pack_qkv_destination_major(q.contiguous(), k.contiguous(), v, world_size)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
