import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn.functional as F

from sglang.multimodal_gen.runtime.layers.attention.layer import (
    UlyssesAttention,
    UlyssesAttention_VSA,
    USPAttention,
    _count_active_replicated_modes,
    _kv_gather_unsupported_reason,
    _resolve_sp_attention_mode,
)
from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum
from sglang.test.test_utils import CustomTestCase

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
    obj.sp_attention_mode_is_auto = False
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
        self.attn.sp_attention_mode_is_auto = False

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


class TestSpAttentionModeResolution(unittest.TestCase):
    def _resolve(self, *, degree=2, auto=True, causal=False, sparse=False):
        stub = SimpleNamespace(kv_gather_degree=degree, sp_split_auto=auto)
        with patch(
            "sglang.multimodal_gen.runtime.server_args.get_global_server_args",
            return_value=stub,
        ):
            return _resolve_sp_attention_mode(causal=causal, sparse_backend=sparse)

    def test_gather_degree_selects_the_gather_exchange(self):
        self.assertEqual(self._resolve(), ("kv_gather", True))
        self.assertEqual(self._resolve(auto=False), ("kv_gather", False))

    def test_degree_one_is_plain_ulysses(self):
        self.assertEqual(self._resolve(degree=1), ("ulysses", False))
        self.assertEqual(self._resolve(degree=1, causal=True), ("ulysses", False))

    def test_auto_degree_falls_back_for_unsupported_layers(self):
        self.assertEqual(self._resolve(causal=True), ("ulysses", True))
        self.assertEqual(self._resolve(sparse=True), ("ulysses", True))

    def test_explicit_degree_fails_closed(self):
        with self.assertRaises(ValueError):
            self._resolve(auto=False, causal=True)
        with self.assertRaises(NotImplementedError):
            self._resolve(auto=False, sparse=True)


class TestKVGatherCallSupport(unittest.TestCase):
    def _reason(self, **overrides):
        kwargs = dict(
            qkv_pre_all_to_all=False,
            replicated_mode_count=0,
            attn_mask=None,
            num_replicated_kv_prefix=0,
        )
        kwargs.update(overrides)
        return _kv_gather_unsupported_reason(**kwargs)

    def test_plain_and_masked_calls_are_supported(self):
        self.assertIsNone(self._reason())
        self.assertIsNone(self._reason(attn_mask=torch.ones(1, 4, dtype=torch.bool)))

    def test_unsupported_shapes_are_reported(self):
        self.assertIn("pre-all-to-all", self._reason(qkv_pre_all_to_all=True))
        self.assertIn("replicated-token", self._reason(replicated_mode_count=2))
        self.assertIn(
            "KV-only prefix",
            self._reason(
                attn_mask=torch.ones(1, 4, dtype=torch.bool),
                num_replicated_kv_prefix=2,
            ),
        )
        self.assertIn("[B, S_local]", self._reason(attn_mask=torch.ones(1, 1, 4)))
        self.assertIn("integer padding", self._reason(attn_mask=torch.ones(1, 4)))

    def test_explicit_mode_raises_and_auto_falls_back_at_dispatch(self):
        attn = _make_attention(4)
        attn.skip_sequence_parallel = False
        q = torch.randn(1, 4, 3, 4)
        with (
            patch(
                f"{_LAYER}.get_forward_context",
                return_value=SimpleNamespace(attn_metadata=None),
            ),
            patch(f"{_LAYER}.get_sequence_parallel_world_size", return_value=2),
        ):
            attn.sp_attention_mode_is_auto = False
            with self.assertRaisesRegex(NotImplementedError, "pre-all-to-all"):
                attn.forward(q, q, q, qkv_pre_all_to_all=True)


class TestReplicatedModeCountCompile(CustomTestCase):
    def test_symbolic_shape_compiles(self):
        def add_mode_count(x):
            count = _count_active_replicated_modes(x.shape[0], 0, 0)
            return x + count

        compiled = torch.compile(
            add_mode_count,
            backend="eager",
            fullgraph=True,
            dynamic=True,
        )
        actual = compiled(torch.ones(3))
        torch.testing.assert_close(actual, torch.full((3,), 2.0))


if __name__ == "__main__":
    unittest.main()
