from unittest import mock

import pytest
import torch
from torch import nn

from sglang.multimodal_gen.runtime.layers.attention import layer as attention_layer
from sglang.multimodal_gen.runtime.models.bridges import mova_dual_tower
from sglang.multimodal_gen.runtime.models.dits import ltx_2
from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum


class _FakeAttentionImpl(nn.Module):
    def __init__(self, **_kwargs) -> None:
        super().__init__()


class _FakeAttentionBackend:
    @classmethod
    def get_enum(cls) -> AttentionBackendEnum:
        return AttentionBackendEnum.FA

    @classmethod
    def get_impl_cls(cls):
        return _FakeAttentionImpl


def test_local_attention_forwards_cross_attention_role():
    with (
        mock.patch.object(
            attention_layer, "get_compute_dtype", return_value=torch.bfloat16
        ),
        mock.patch.object(
            attention_layer, "get_attn_backend", return_value=_FakeAttentionBackend
        ) as get_backend,
        mock.patch.object(attention_layer, "wrap_attention_impl_forward"),
    ):
        attention_layer.LocalAttention(
            num_heads=1,
            head_size=64,
            is_cross_attention=True,
        )

    assert get_backend.call_args.kwargs["is_cross_attention"] is True


@pytest.mark.parametrize("use_local_attention", [False, True])
def test_ltx2_derives_cross_attention_role_from_context(use_local_attention):
    selected_layer = "LocalAttention" if use_local_attention else "USPAttention"
    with (
        mock.patch.object(ltx_2, "get_tp_world_size", return_value=1),
        mock.patch.object(ltx_2, "ColumnParallelLinear", return_value=nn.Identity()),
        mock.patch.object(ltx_2, "RowParallelLinear", return_value=nn.Identity()),
        mock.patch.object(ltx_2, selected_layer) as attention,
    ):
        ltx_2.LTX2Attention(
            query_dim=8,
            context_dim=8,
            heads=1,
            dim_head=8,
            use_local_attention=use_local_attention,
        )
        cross_attention_kwargs = attention.call_args.kwargs
        attention.reset_mock()
        ltx_2.LTX2Attention(
            query_dim=8,
            heads=1,
            dim_head=8,
            use_local_attention=use_local_attention,
        )
        self_attention_kwargs = attention.call_args.kwargs

    assert cross_attention_kwargs["is_cross_attention"] is True
    assert self_attention_kwargs["is_cross_attention"] is False


def test_mova_bridge_marks_conditional_attention_as_cross_attention():
    with (
        mock.patch.object(mova_dual_tower, "get_tp_world_size", return_value=1),
        mock.patch.object(
            mova_dual_tower, "ColumnParallelLinear", return_value=nn.Identity()
        ),
        mock.patch.object(
            mova_dual_tower, "RowParallelLinear", return_value=nn.Identity()
        ),
        mock.patch.object(mova_dual_tower, "USPAttention") as attention,
    ):
        mova_dual_tower.ConditionalCrossAttention(
            dim=8,
            kv_dim=8,
            num_heads=1,
        )

    assert attention.call_args.kwargs["is_cross_attention"] is True
