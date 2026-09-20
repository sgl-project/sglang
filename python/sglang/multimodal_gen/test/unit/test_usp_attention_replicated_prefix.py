"""Regression test for USPAttention GQA replicated-prefix head sharding.

``USPAttention._forward_with_replicated_prefix`` keeps a replicated token prefix
(e.g. text) out of the Ulysses all-to-all and slices that prefix down to the local
head shard. For a GQA model (kv heads < q heads) the K/V prefix must be sliced by
the *KV* head shard, not the query head shard -- otherwise the per-rank query
offset overshoots the KV head dim, the prefix slice is empty/mismatched, and the
``cat`` with the all-to-all'd suffix raises. MHA (kv heads == q heads) is unaffected.

Single-process test: the Ulysses world size, rank, all-to-all helpers, and
all_gather are mocked so the per-rank slicing logic runs on CPU.
"""

import unittest
from unittest.mock import MagicMock, patch

import torch

from sglang.multimodal_gen.runtime.layers.attention.layer import USPAttention
from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum

_LAYER = "sglang.multimodal_gen.runtime.layers.attention.layer"
_SP = 2


def _fake_input_all_to_all(x, **_):
    # Ulysses input all-to-all: gather sequence (xSP), shard heads (/SP). Only the
    # resulting shape matters for this slicing test.
    h = x.shape[2]
    return x[:, :, : h // _SP, :].repeat_interleave(_SP, dim=1).contiguous()


def _fake_output_all_to_all(x, **_):
    # Inverse of the input all-to-all: shard sequence (/SP), gather heads (xSP).
    s = x.shape[1]
    return x[:, : s // _SP, :, :].repeat_interleave(_SP, dim=2).contiguous()


class _CaptureAttn:
    """Stand-in attn backend that records the q/k/v it receives."""

    def __init__(self):
        self.q = self.k = self.v = None

    def forward(self, q, k, v, _ctx):
        self.q, self.k, self.v = q, k, v
        return q.clone()


class TestUSPAttentionReplicatedPrefix(unittest.TestCase):
    def _run(self, q_heads, kv_heads, sp_rank, num_rep=3, suffix=4, head_dim=4):
        attn = _CaptureAttn()
        obj = USPAttention.__new__(USPAttention)  # bypass __init__/backend setup
        obj.attn_impl = attn

        seq = num_rep + suffix
        q = torch.randn(1, seq, q_heads, head_dim)
        k = torch.randn(1, seq, kv_heads, head_dim)
        v = torch.randn(1, seq, kv_heads, head_dim)

        sp_group = MagicMock()
        sp_group.ulysses_group = None

        def fake_all_gather(out_list, tensor, **_):
            for t in out_list:
                t.copy_(tensor)

        with (
            patch(f"{_LAYER}.get_ulysses_parallel_world_size", return_value=_SP),
            patch(f"{_LAYER}.get_ulysses_parallel_rank", return_value=sp_rank),
            patch(f"{_LAYER}.get_ring_parallel_world_size", return_value=1),
            patch(
                f"{_LAYER}._usp_input_all_to_all", side_effect=_fake_input_all_to_all
            ),
            patch(
                f"{_LAYER}._usp_output_all_to_all",
                side_effect=_fake_output_all_to_all,
            ),
            patch(f"{_LAYER}.get_sp_group", return_value=sp_group),
            patch("torch.distributed.all_gather", side_effect=fake_all_gather),
        ):
            out = USPAttention._forward_with_replicated_prefix(
                obj, q, k, v, None, num_rep
            )
        return attn, out, q.shape

    def test_gqa_slices_kv_prefix_by_kv_heads(self):
        # GQA: 8 query heads, 2 kv heads. The old code sliced the K/V prefix by the
        # query head shard, producing an empty/mismatched prefix and a cat error.
        for sp_rank in range(_SP):
            with self.subTest(sp_rank=sp_rank):
                attn, out, q_shape = self._run(q_heads=8, kv_heads=2, sp_rank=sp_rank)
                # q keeps q_heads/SP, k/v keep kv_heads/SP -> GQA grouping preserved.
                self.assertEqual(attn.q.shape[2], 8 // _SP)
                self.assertEqual(attn.k.shape[2], 2 // _SP)
                self.assertEqual(attn.v.shape[2], 2 // _SP)
                # prefix + all-to-all'd suffix line up on the sequence axis.
                self.assertEqual(attn.k.shape[1], attn.q.shape[1])
                # output is restored to the input layout.
                self.assertEqual(tuple(out.shape), tuple(q_shape))

    def test_mha_prefix_unchanged(self):
        # MHA: q heads == kv heads, so the KV-shard slicing is identical to before.
        attn, out, q_shape = self._run(q_heads=8, kv_heads=8, sp_rank=1)
        self.assertEqual(attn.q.shape[2], 8 // _SP)
        self.assertEqual(attn.k.shape[2], 8 // _SP)
        self.assertEqual(tuple(out.shape), tuple(q_shape))


if __name__ == "__main__":
    unittest.main()


class TestUSPAttentionMaskedReplicatedGuard(unittest.TestCase):
    def test_masked_path_rejects_replicated_tokens(self):
        obj = USPAttention.__new__(USPAttention)
        obj.attn_impl = _CaptureAttn()
        obj.skip_sequence_parallel = False
        obj.sp_attention_mode = "ulysses"
        obj.sp_attention_mode_is_auto = False
        q = torch.randn(1, 6, 2, 4)
        mask = torch.ones(1, 6, dtype=torch.bool)
        with (
            patch(
                f"{_LAYER}.get_forward_context",
                return_value=MagicMock(attn_metadata=None),
            ),
            patch(f"{_LAYER}.get_sequence_parallel_world_size", return_value=2),
        ):
            with self.assertRaisesRegex(NotImplementedError, "replicated"):
                obj.forward(q, q, q, attn_mask=mask, num_replicated_suffix=2)

    def test_single_rank_masked_call_keeps_replicated_args(self):
        # Without SP the mask describes the full sequence; the replicated
        # counts are meaningless and must not be refused.
        obj = USPAttention.__new__(USPAttention)
        obj.attn_impl = _CaptureAttn()
        obj.skip_sequence_parallel = False
        obj.sp_attention_mode = "ulysses"
        obj.sp_attention_mode_is_auto = False
        obj.allow_cudnn_sdp = False
        obj.softmax_scale = 0.5
        obj.backend = AttentionBackendEnum.TORCH_SDPA
        obj.causal = False
        obj.dropout_p = 0.0
        q = torch.randn(1, 6, 2, 4)
        mask = torch.ones(1, 6, dtype=torch.bool)
        with (
            patch(
                f"{_LAYER}.get_forward_context",
                return_value=MagicMock(attn_metadata=None),
            ),
            patch(f"{_LAYER}.get_sequence_parallel_world_size", return_value=1),
        ):
            out = obj.forward(q, q, q, attn_mask=mask, num_replicated_prefix=2)
        self.assertEqual(out.shape, q.shape)


class TestUSPAttentionMaskedReplicatedPrefix(unittest.TestCase):
    """A [B, S] key mask with a replicated prefix now rides the
    replicated-prefix flow under ulysses SP instead of raising: after the
    all-to-all the attention is rank-local over the full sequence, so the
    output must match a plain full-sequence masked SDPA reference."""

    _H, _D = 4, 8
    _NUM_REP, _S_LOCAL, _S_OTHER = 3, 4, 4

    def _run_rank0(self, q_full, k_full, v_full, text_mask, other_suffix_mask):

        num_rep, s_local = self._NUM_REP, self._S_LOCAL
        h_local = self._H // _SP
        obj = USPAttention.__new__(USPAttention)
        obj.attn_impl = _CaptureAttn()
        obj.skip_sequence_parallel = False
        obj.sp_attention_mode = "ulysses"
        obj.sp_attention_mode_is_auto = True
        obj.backend = None  # not FA -> the masked branch takes SDPA
        obj.softmax_scale = self._D**-0.5
        obj.allow_cudnn_sdp = False
        obj.enable_packed_qkv_input_a2a = False

        local = slice(0, num_rep + s_local)
        q, k, v = q_full[:, local], k_full[:, local], v_full[:, local]
        other = {
            "suffix": lambda full: full[:, num_rep + s_local :],
        }
        gathered = iter(
            [
                torch.cat([q[:, num_rep:], other["suffix"](q_full)], dim=1),
                torch.cat([k[:, num_rep:], other["suffix"](k_full)], dim=1),
                torch.cat([v[:, num_rep:], other["suffix"](v_full)], dim=1),
            ]
        )

        def fake_input_a2a(x, **_):
            # rank0 view: gather sequence across ranks, keep the first head shard.
            return next(gathered)[:, :, :h_local, :].contiguous()

        def fake_output_a2a(x, **_):
            # inverse: keep rank0's sequence shard, duplicate the head shard.
            return x[:, :s_local].repeat(1, 1, _SP, 1)

        def fake_mask_gather(m, **_):
            return torch.cat([m, other_suffix_mask], dim=1)

        def fake_all_gather(out_list, tensor, **_):
            for t in out_list:
                t.copy_(tensor)

        sp_group = MagicMock()
        sp_group.ulysses_group = None
        local_mask = torch.cat(
            [text_mask, torch.ones(1, s_local, dtype=torch.bool)], dim=1
        )
        with (
            patch(
                f"{_LAYER}.get_forward_context",
                return_value=MagicMock(attn_metadata=None),
            ),
            patch(f"{_LAYER}.get_sequence_parallel_world_size", return_value=_SP),
            patch(f"{_LAYER}.get_ulysses_parallel_world_size", return_value=_SP),
            patch(f"{_LAYER}.get_ring_parallel_world_size", return_value=1),
            patch(f"{_LAYER}.get_ulysses_parallel_rank", return_value=0),
            patch(f"{_LAYER}._usp_input_all_to_all", side_effect=fake_input_a2a),
            patch(f"{_LAYER}._usp_output_all_to_all", side_effect=fake_output_a2a),
            patch(
                f"{_LAYER}.sequence_model_parallel_all_gather",
                side_effect=fake_mask_gather,
            ),
            patch(f"{_LAYER}.get_sp_group", return_value=sp_group),
            patch("torch.distributed.all_gather", side_effect=fake_all_gather),
        ):
            return obj.forward(
                q, k, v, attn_mask=local_mask, num_replicated_prefix=num_rep
            )

    def test_matches_full_sequence_masked_sdpa(self):
        torch.manual_seed(0)
        seq = self._NUM_REP + self._S_LOCAL + self._S_OTHER
        q = torch.randn(1, seq, self._H, self._D)
        k = torch.randn(1, seq, self._H, self._D)
        v = torch.randn(1, seq, self._H, self._D)
        text_mask = torch.tensor([[True, True, False]])
        other_suffix_mask = torch.ones(1, self._S_OTHER, dtype=torch.bool)

        out = self._run_rank0(q, k, v, text_mask, other_suffix_mask)

        full_mask = torch.cat(
            [text_mask, torch.ones(1, self._S_LOCAL + self._S_OTHER, dtype=torch.bool)],
            dim=1,
        )
        ref = torch.nn.functional.scaled_dot_product_attention(
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
            attn_mask=full_mask[:, None, None, :],
            scale=self._D**-0.5,
        ).transpose(1, 2)

        h_local = self._H // _SP
        rows = self._NUM_REP + self._S_LOCAL
        self.assertEqual(tuple(out.shape), (1, rows, self._H, self._D))
        torch.testing.assert_close(
            out[:, :, :h_local, :],
            ref[:, :rows, :h_local, :],
            rtol=1e-5,
            atol=1e-5,
        )

    def test_masked_prefix_still_rejects_ring_and_3d_masks(self):
        obj = USPAttention.__new__(USPAttention)
        obj.attn_impl = _CaptureAttn()
        obj.skip_sequence_parallel = False
        obj.sp_attention_mode = "ulysses"
        obj.sp_attention_mode_is_auto = True
        q = torch.randn(1, 6, 2, 4)
        with (
            patch(
                f"{_LAYER}.get_forward_context",
                return_value=MagicMock(attn_metadata=None),
            ),
            patch(f"{_LAYER}.get_sequence_parallel_world_size", return_value=2),
            patch(f"{_LAYER}.get_ring_parallel_world_size", return_value=2),
        ):
            with self.assertRaisesRegex(NotImplementedError, "replicated"):
                obj.forward(
                    q,
                    q,
                    q,
                    attn_mask=torch.ones(1, 6, dtype=torch.bool),
                    num_replicated_prefix=2,
                )
        with (
            patch(
                f"{_LAYER}.get_forward_context",
                return_value=MagicMock(attn_metadata=None),
            ),
            patch(f"{_LAYER}.get_sequence_parallel_world_size", return_value=2),
            patch(f"{_LAYER}.get_ring_parallel_world_size", return_value=1),
        ):
            with self.assertRaisesRegex(NotImplementedError, "replicated"):
                obj.forward(
                    q,
                    q,
                    q,
                    attn_mask=torch.ones(1, 6, 6, dtype=torch.bool),
                    num_replicated_prefix=2,
                )
