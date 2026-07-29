import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn.functional as F

from sglang.multimodal_gen.runtime.layers.attention.layer import (
    UlyssesAttention,
    UlyssesAttention_VSA,
    USPAttention,
)
from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum

_LAYER = "sglang.multimodal_gen.runtime.layers.attention.layer"


class _SdpaAttention:
    def __init__(self, scale: float):
        self.scale = scale

    def forward(self, q, k, v, _ctx):
        return F.scaled_dot_product_attention(
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
            dropout_p=0.0,
            is_causal=False,
            scale=self.scale,
        ).transpose(1, 2)


def _make_attention(head_dim: int) -> USPAttention:
    obj = USPAttention.__new__(USPAttention)
    obj.causal = False
    obj.backend = AttentionBackendEnum.TORCH_SDPA
    obj.softmax_scale = head_dim**-0.5
    obj.attn_impl = _SdpaAttention(obj.softmax_scale)
    obj.allow_cudnn_sdp = False
    obj.skip_sequence_parallel = False
    obj.sp_attention_mode = "kv_gather"
    return obj


def _reference_attention(q, k, v, scale, key_mask=None, query_mask=None):
    out = F.scaled_dot_product_attention(
        q.transpose(1, 2),
        k.transpose(1, 2),
        v.transpose(1, 2),
        attn_mask=None if key_mask is None else key_mask[:, None, None, :],
        dropout_p=0.0,
        is_causal=False,
        scale=scale,
    ).transpose(1, 2)
    if query_mask is not None:
        out = out * query_mask[:, :, None, None]
    return out


class TestUSPAttentionKVGather(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.heads = 3
        self.head_dim = 4
        self.attn = _make_attention(self.head_dim)

    def _run(self, q, k, v, gathered, **kwargs):
        with (
            patch(f"{_LAYER}.get_ring_parallel_world_size", return_value=1),
            patch(
                f"{_LAYER}.sequence_model_parallel_all_gather",
                side_effect=gathered,
            ),
        ):
            return self.attn._forward_with_kv_gather(
                q,
                k,
                v,
                None,
                kwargs.pop("attn_mask", None),
                kwargs.pop("attn_mask_meta", None),
                kwargs.pop("num_replicated_prefix", 0),
                kwargs.pop("num_replicated_suffix", 0),
                kwargs.pop("num_replicated_kv_prefix", 0),
            )

    def test_local_queries_attend_gathered_kv(self):
        q = torch.randn(1, 3, self.heads, self.head_dim)
        k = torch.randn(1, 3, self.heads, self.head_dim)
        v = torch.randn(1, 3, self.heads, self.head_dim)
        full_k = torch.randn(1, 6, self.heads, self.head_dim)
        full_v = torch.randn(1, 6, self.heads, self.head_dim)

        out = self._run(q, k, v, [full_k, full_v])
        expected = _reference_attention(q, full_k, full_v, self.attn.softmax_scale)

        torch.testing.assert_close(out, expected)

    def test_replicated_prefix_is_not_duplicated(self):
        prefix = 2
        q = torch.randn(1, 4, self.heads, self.head_dim)
        k = torch.randn(1, 4, self.heads, self.head_dim)
        v = torch.randn(1, 4, self.heads, self.head_dim)
        gathered_k_suffix = torch.randn(1, 4, self.heads, self.head_dim)
        gathered_v_suffix = torch.randn(1, 4, self.heads, self.head_dim)
        full_k = torch.cat([k[:, :prefix], gathered_k_suffix], dim=1)
        full_v = torch.cat([v[:, :prefix], gathered_v_suffix], dim=1)

        out = self._run(
            q,
            k,
            v,
            [gathered_k_suffix, gathered_v_suffix],
            num_replicated_prefix=prefix,
        )
        expected = _reference_attention(q, full_k, full_v, self.attn.softmax_scale)

        torch.testing.assert_close(out, expected)

    def test_padding_mask_uses_local_queries_and_global_keys(self):
        q = torch.randn(2, 3, self.heads, self.head_dim)
        k = torch.randn(2, 3, self.heads, self.head_dim)
        v = torch.randn(2, 3, self.heads, self.head_dim)
        full_k = torch.randn(2, 6, self.heads, self.head_dim)
        full_v = torch.randn(2, 6, self.heads, self.head_dim)
        query_mask = torch.tensor([[True, True, False], [True, True, True]])
        key_mask = torch.tensor(
            [
                [True, True, False, True, False, False],
                [True, True, True, True, True, False],
            ]
        )

        out = self._run(
            q,
            k,
            v,
            [full_k, full_v, key_mask],
            attn_mask=query_mask,
            attn_mask_meta={},
        )
        expected = _reference_attention(
            q,
            full_k,
            full_v,
            self.attn.softmax_scale,
            key_mask=key_mask,
            query_mask=query_mask,
        )

        torch.testing.assert_close(out, expected)

    def test_separate_replicated_kv_prefix_gathers_only_suffix(self):
        q = torch.randn(1, 3, self.heads, self.head_dim)
        k_prefix = torch.randn(1, 2, self.heads, self.head_dim)
        v_prefix = torch.randn(1, 2, self.heads, self.head_dim)
        k_suffix = torch.randn(1, 3, self.heads, self.head_dim)
        v_suffix = torch.randn(1, 3, self.heads, self.head_dim)
        gathered_k_suffix = torch.randn(1, 6, self.heads, self.head_dim)
        gathered_v_suffix = torch.randn(1, 6, self.heads, self.head_dim)
        full_k = torch.cat([k_prefix, gathered_k_suffix], dim=1)
        full_v = torch.cat([v_prefix, gathered_v_suffix], dim=1)

        with (
            patch(
                f"{_LAYER}.get_forward_context",
                return_value=SimpleNamespace(attn_metadata=None),
            ),
            patch(f"{_LAYER}.get_sequence_parallel_world_size", return_value=2),
            patch(f"{_LAYER}.get_ring_parallel_world_size", return_value=1),
            patch(
                f"{_LAYER}.sequence_model_parallel_all_gather",
                side_effect=[gathered_k_suffix, gathered_v_suffix],
            ),
        ):
            out = self.attn.forward_with_replicated_kv_prefix(
                q, k_prefix, v_prefix, k_suffix, v_suffix
            )
        expected = _reference_attention(q, full_k, full_v, self.attn.softmax_scale)

        torch.testing.assert_close(out, expected)


class TestUlyssesAttentionKVGather(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1)
        self.heads = 3
        self.head_dim = 4
        self.attn = UlyssesAttention.__new__(UlyssesAttention)
        self.attn.causal = False
        self.attn.backend = AttentionBackendEnum.TORCH_SDPA
        self.attn.softmax_scale = self.head_dim**-0.5
        self.attn.attn_impl = _SdpaAttention(self.attn.softmax_scale)
        self.attn.sp_attention_mode = "kv_gather"

    def _run(self, q, k, v, gathered, **kwargs):
        with (
            patch(
                f"{_LAYER}.get_forward_context",
                return_value=SimpleNamespace(attn_metadata=None),
            ),
            patch(f"{_LAYER}.get_ring_parallel_world_size", return_value=1),
            patch(
                f"{_LAYER}.sequence_model_parallel_all_gather",
                side_effect=gathered,
            ),
        ):
            return self.attn.forward(
                q,
                k,
                v,
                kwargs.get("replicated_q"),
                kwargs.get("replicated_k"),
                kwargs.get("replicated_v"),
                kwargs.get("seq_lens"),
            )

    def test_local_queries_attend_gathered_kv(self):
        q = torch.randn(1, 3, self.heads, self.head_dim)
        k = torch.randn(1, 3, self.heads, self.head_dim)
        v = torch.randn(1, 3, self.heads, self.head_dim)
        full_k = torch.randn(1, 6, self.heads, self.head_dim)
        full_v = torch.randn(1, 6, self.heads, self.head_dim)

        out, replicated_out = self._run(q, k, v, [full_k, full_v])
        expected = _reference_attention(q, full_k, full_v, self.attn.softmax_scale)

        torch.testing.assert_close(out, expected)
        self.assertIsNone(replicated_out)

    def test_replicated_suffix_is_computed_without_head_sharding(self):
        q = torch.randn(1, 3, self.heads, self.head_dim)
        k = torch.randn(1, 3, self.heads, self.head_dim)
        v = torch.randn(1, 3, self.heads, self.head_dim)
        replicated_q = torch.randn(1, 2, self.heads, self.head_dim)
        replicated_k = torch.randn(1, 2, self.heads, self.head_dim)
        replicated_v = torch.randn(1, 2, self.heads, self.head_dim)
        full_k = torch.randn(1, 6, self.heads, self.head_dim)
        full_v = torch.randn(1, 6, self.heads, self.head_dim)

        out, replicated_out = self._run(
            q,
            k,
            v,
            [full_k, full_v],
            replicated_q=replicated_q,
            replicated_k=replicated_k,
            replicated_v=replicated_v,
        )
        full_q = torch.cat([q, replicated_q], dim=1)
        full_k = torch.cat([full_k, replicated_k], dim=1)
        full_v = torch.cat([full_v, replicated_v], dim=1)
        expected = _reference_attention(full_q, full_k, full_v, self.attn.softmax_scale)

        torch.testing.assert_close(out, expected[:, : q.shape[1]])
        torch.testing.assert_close(replicated_out, expected[:, q.shape[1] :])

    def test_varlen_is_rejected(self):
        q = torch.randn(1, 3, self.heads, self.head_dim)
        with self.assertRaisesRegex(NotImplementedError, "varlen"):
            self._run(q, q, q, [], seq_lens=[3, 3])

    def test_video_sparse_attention_is_rejected(self):
        attn = UlyssesAttention_VSA.__new__(UlyssesAttention_VSA)
        attn.sp_attention_mode = "kv_gather"
        q = torch.randn(1, 3, self.heads, self.head_dim)
        with self.assertRaisesRegex(NotImplementedError, "video sparse"):
            attn.forward(q, q, q, gate_compress=q)


if __name__ == "__main__":
    unittest.main()
