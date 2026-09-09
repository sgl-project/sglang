"""Replicated-token paths under ring parallelism: dispatch, merge, KV order."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.multimodal_gen.runtime.layers.attention.layer import USPAttention
from sglang.multimodal_gen.runtime.layers.usp import _merge_attention_partials
from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum

_LAYER = "sglang.multimodal_gen.runtime.layers.attention.layer"


def _sdpa(q, k, v, scale):
    return torch.nn.functional.scaled_dot_product_attention(
        q.transpose(1, 2).float(),
        k.transpose(1, 2).float(),
        v.transpose(1, 2).float(),
        scale=scale,
    ).transpose(1, 2)


class _LseImpl:
    """SDPA with explicit LSE so merge math can be verified for real."""

    def __init__(self, scale):
        self.scale = scale
        self.seen_k = []

    def forward(self, q, k, v, attn_metadata=None, return_softmax_lse=False):
        self.seen_k.append(k)
        out = _sdpa(q, k, v, self.scale)
        if not return_softmax_lse:
            return out.to(q.dtype)
        logits = torch.einsum("bshd,bthd->bhst", q.float(), k.float()) * self.scale
        lse = torch.logsumexp(logits, dim=-1)  # [B, H, S]
        return out, lse


def _ring_pair_via_impl(impl, q, k_shard, v_shard):
    """Stand-in for ring_attn on a 1-chunk ring: one local partial + LSE."""
    return impl.forward(
        q, k_shard, v_shard, attn_metadata=None, return_softmax_lse=True
    )


class RingReplicatedBase(unittest.TestCase):
    B, S_SHARD, REP, H, D = 1, 6, 3, 2, 8

    def _attn(self):
        obj = USPAttention.__new__(USPAttention)
        obj.skip_sequence_parallel = False
        obj.sp_attention_mode = "ulysses"
        obj.sp_attention_mode_is_auto = False
        obj.softmax_scale = self.D**-0.5
        obj.backend = AttentionBackendEnum.FA
        obj.causal = False
        obj.dropout_p = 0.0
        obj.attn_impl = _LseImpl(obj.softmax_scale)
        return obj

    def _patches(self, ring_ws=2):
        return (
            patch(
                f"{_LAYER}.get_forward_context",
                return_value=SimpleNamespace(attn_metadata=None),
            ),
            patch(f"{_LAYER}.get_sequence_parallel_world_size", return_value=ring_ws),
            patch(f"{_LAYER}.get_ulysses_parallel_world_size", return_value=1),
            patch(f"{_LAYER}.get_ring_parallel_world_size", return_value=ring_ws),
            patch(f"{_LAYER}.get_ulysses_parallel_rank", return_value=0),
            patch(f"{_LAYER}._usp_input_all_to_all", side_effect=lambda x, head_dim: x),
            patch(
                f"{_LAYER}._usp_output_all_to_all", side_effect=lambda x, head_dim: x
            ),
            patch(
                f"{_LAYER}.ring_attn",
                side_effect=lambda q, k, v, impl, return_softmax_lse: (
                    _ring_pair_via_impl(impl, q, k, v)
                ),
            ),
        )

    def _rand(self, s):
        return torch.randn(self.B, s, self.H, self.D)


class TestRingReplicatedPrefix(RingReplicatedBase):
    def test_u1_r2_dispatches_and_merges_exactly(self):
        obj = self._attn()
        q = self._rand(self.REP + self.S_SHARD)
        k = self._rand(self.REP + self.S_SHARD)
        v = self._rand(self.REP + self.S_SHARD)

        ps = self._patches()
        with ps[0], ps[1], ps[2], ps[3], ps[4], ps[5], ps[6], ps[7]:
            out = obj.forward(q, k, v, num_replicated_prefix=self.REP)

        # One local ring chunk + rep partial merged == full attention.
        ref = _sdpa(q, k, v, obj.softmax_scale)
        self.assertEqual(out.shape, q.shape)
        torch.testing.assert_close(out.float(), ref, atol=1e-5, rtol=1e-5)

    def test_kv_prefix_u1_r2_matches_full_attention(self):
        obj = self._attn()
        q = self._rand(self.S_SHARD)
        k = self._rand(self.REP + self.S_SHARD)
        v = self._rand(self.REP + self.S_SHARD)

        ps = self._patches()
        with ps[0], ps[1], ps[2], ps[3], ps[4], ps[5], ps[6], ps[7]:
            out = obj.forward(q, k, v, num_replicated_kv_prefix=self.REP)

        ref = _sdpa(q, k, v, obj.softmax_scale)
        torch.testing.assert_close(out.float(), ref, atol=1e-5, rtol=1e-5)


class TestRingReplicatedSuffix(RingReplicatedBase):
    def test_u1_r2_dispatches_and_merges_exactly(self):
        obj = self._attn()
        q = self._rand(self.S_SHARD + self.REP)
        k = self._rand(self.S_SHARD + self.REP)
        v = self._rand(self.S_SHARD + self.REP)

        ps = self._patches()
        with ps[0], ps[1], ps[2], ps[3], ps[4], ps[5], ps[6], ps[7]:
            out = obj.forward(q, k, v, num_replicated_suffix=self.REP)

        ref = _sdpa(q, k, v, obj.softmax_scale)
        torch.testing.assert_close(out.float(), ref, atol=1e-5, rtol=1e-5)

    def test_non_ring_path_keeps_kv_tail_order(self):
        obj = self._attn()
        q = self._rand(self.S_SHARD + self.REP)
        k = self._rand(self.S_SHARD + self.REP)
        v = self._rand(self.S_SHARD + self.REP)

        def _fake_gather(out_list, t, group=None):
            for o in out_list:
                o.copy_(t)

        ps = self._patches(ring_ws=1)
        with (
            ps[0],
            ps[4],
            ps[5],
            ps[6],
            patch(f"{_LAYER}.get_ring_parallel_world_size", return_value=1),
            patch(f"{_LAYER}.get_sequence_parallel_world_size", return_value=2),
            patch(f"{_LAYER}.get_ulysses_parallel_world_size", return_value=2),
            patch(
                f"{_LAYER}.get_sp_group",
                return_value=SimpleNamespace(ulysses_group=None),
            ),
            patch("torch.distributed.all_gather", side_effect=_fake_gather),
        ):
            # Identity-mocked collectives don't reproduce head-shard shapes,
            # so the final concat may fail — the kernel K order is recorded
            # before that and is all this test asserts.
            try:
                obj.forward(q, k, v, num_replicated_suffix=self.REP)
            except RuntimeError:
                pass

        # Bitwise contract: suffix KV stays at the tail in the kernel call.
        kernel_k = obj.attn_impl.seen_k[-1]
        torch.testing.assert_close(kernel_k[:, -self.REP :], k[:, -self.REP :])


class TestMergePartials(unittest.TestCase):
    def test_two_disjoint_halves_merge_to_full_attention(self):
        torch.manual_seed(7)
        B, S, T, H, D = 1, 5, 8, 2, 16
        scale = D**-0.5
        q = torch.randn(B, S, H, D)
        k = torch.randn(B, T, H, D)
        v = torch.randn(B, T, H, D)

        def part(ks, vs):
            logits = torch.einsum("bshd,bthd->bhst", q, ks) * scale
            lse = torch.logsumexp(logits, dim=-1)
            out = torch.softmax(logits, dim=-1)
            return torch.einsum("bhst,bthd->bshd", out, vs), lse

        out_a, lse_a = part(k[:, :3], v[:, :3])
        out_b, lse_b = part(k[:, 3:], v[:, 3:])
        merged = _merge_attention_partials(out_a, lse_a, out_b, lse_b)

        ref, _ = part(k, v)
        torch.testing.assert_close(merged, ref, atol=1e-6, rtol=1e-6)


if __name__ == "__main__":
    unittest.main()
