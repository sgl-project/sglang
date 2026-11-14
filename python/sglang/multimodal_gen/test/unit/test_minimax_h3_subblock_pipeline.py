# SPDX-License-Identifier: Apache-2.0
"""H3 mask wiring only; communication and attention kernels are mocked."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from sglang.multimodal_gen.runtime.distributed.device_communicators import ipc_a2a_multi
from sglang.multimodal_gen.runtime.models.dits import minimax_h3 as h3
from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum


@pytest.mark.parametrize("fused", [False, True])
@pytest.mark.parametrize(
    "backend", [AttentionBackendEnum.SUBBLOCK_SPARSE_ATTN, AttentionBackendEnum.FA]
)
def test_pipeline_mask_forwarding(monkeypatch, backend, fused):
    monkeypatch.setattr(h3.envs, "SGLANG_DIFFUSION_ULYSSES_PIPELINE_GROUPS", 7)
    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: False)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setattr(h3, "_usp_input_all_to_all_packed_qkv", lambda *qkv: qkv)
    monkeypatch.setattr(h3, "_usp_output_all_to_all", lambda out, **kw: out)
    monkeypatch.setattr(h3, "fused_inplace_qknorm_rope", Mock())
    impl = SimpleNamespace(forward_varlen=Mock(side_effect=lambda q, k, v, **kw: q))
    norm = SimpleNamespace(weight=torch.ones(4), eps=1e-5)
    attention = SimpleNamespace(
        _attention_impl=impl,
        _attention_backend_enum=backend,
        _use_fused_qknorm_rope=True,
        num_heads=2,
        head_dim=4,
        q_norm=norm,
        k_norm=norm,
    )
    qkv = [torch.zeros(12, 2, 4, dtype=torch.bfloat16) for _ in range(3)]
    mask = torch.tensor([True, False])
    kwargs = dict(
        cu_seqlens=torch.tensor([0, 8, 12], dtype=torch.int32),
        cu_seqlens_host=(0, 8, 12),
        max_seqlen=8,
    )

    def pipeline(q, k, v, attend, groups, **options):
        assert groups == 7
        assert (options["fill"] is not None) == fused
        options["sequential"]()
        return attend(q, k, v)

    transport = Mock(side_effect=pipeline)
    monkeypatch.setattr(ipc_a2a_multi, "ulysses_pipelined_attention", transport)
    if fused:
        h3.MiniMaxH3Attention._attend(
            attention,
            torch.zeros(12, 8),
            *qkv,
            **kwargs,
            rope_cache=(torch.zeros(12, 4), torch.arange(12)),
            subblock_sparse_query_block_mask=mask,
            ulysses_active=True,
            ring_active=False,
        )
    else:
        h3._minimax_h3_attention_core_impl(
            attention,
            *qkv,
            **kwargs,
            ulysses_active=True,
            subblock_sparse_query_block_mask=mask,
        )
    expected = dict(kwargs)
    if backend is AttentionBackendEnum.SUBBLOCK_SPARSE_ATTN:
        expected["first_segment_sparse_query_block_mask"] = mask
    transport.assert_called_once()
    assert impl.forward_varlen.call_count == 2  # serial reference and group callback
    assert all(call.kwargs == expected for call in impl.forward_varlen.call_args_list)


@pytest.mark.parametrize("sparse_enabled", [False, True])
def test_missing_mask_only_errors_for_sparse(sparse_enabled):
    q = torch.zeros(8, 1, 4, dtype=torch.bfloat16)
    impl = SimpleNamespace(
        schedule=SimpleNamespace(min_seq_len=4),
        _sparse_ready=lambda q, k: sparse_enabled,
        forward_varlen=Mock(return_value=q),
    )
    attend = h3._minimax_h3_group_attend(
        impl,
        cu_seqlens=torch.tensor([0, 8], dtype=torch.int32),
        cu_seqlens_host=(0, 8),
        max_seqlen=8,
        first_segment_sparse_query_block_mask=None,
    )
    if sparse_enabled:
        with pytest.raises(
            ValueError, match="requires subblock_sparse_query_block_mask"
        ):
            attend(q, q, q)
        impl.forward_varlen.assert_not_called()
    else:
        assert attend(q, q, q) is q
