"""Unit tests for QKV LoRA correctness when ``tp_size > num_key_value_heads``.

Covers the case where :class:`QKVParallelLinear` *replicates* each KV head
across ``tp_size // num_kv_heads`` ranks instead of further dividing them.
Two cooperating pieces have to agree on the per-rank QKV layout:

1. :meth:`LoRAMemoryPool._column_parallel_lora_b_per_rank_dim` must size
   the LoRA B buffer to the actual per-rank QKV output (``q_per_rank +
   2 * head_dim``), not a naive ``divide(total, tp_size)``.

2. :meth:`QKVParallelLinearWithLoRA.slice_lora_b_weights` must index the
   PEFT-format (un-replicated) ``B`` tensor using un-replicated K/V
   sizes — otherwise the V slice falls past the end of ``B`` and silently
   returns 0 rows, tripping the buffer-shape assert in the LoRA mem pool.

Tests are hermetic: ``LoRAMemoryPool`` and ``QKVParallelLinearWithLoRA``
are constructed via ``__new__`` so we don't need a real base model,
distributed groups, or CUDA.

Usage:
    python -m pytest test/registered/unit/lora/test_qkv_lora_kv_replication.py -v
"""

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=5, suite="stage-b-test-1-gpu-small")

import types
import unittest

import torch

from sglang.srt.lora.layers import QKVParallelLinearWithLoRA
from sglang.srt.lora.mem_pool import LoRAMemoryPool


def _make_pool(
    *, head_dim: int, num_attention_heads: int, num_kv_heads: int
) -> LoRAMemoryPool:
    """Build a minimal pool whose `base_hf_config` carries just enough fields
    for `_column_parallel_lora_b_per_rank_dim` to do its job."""
    pool = LoRAMemoryPool.__new__(LoRAMemoryPool)
    pool.base_hf_config = types.SimpleNamespace(
        head_dim=head_dim,
        hidden_size=num_attention_heads * head_dim,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_kv_heads,
    )
    return pool


def _qkv_total_output_dim(
    *,
    head_dim: int,
    num_attention_heads: int,
    num_kv_heads: int,
    attn_output_gate: bool,
) -> int:
    """Mirror the per-model `get_hidden_dim("qkv_proj")` return: the
    un-sharded total, optionally gate-doubled on the q side."""
    q_heads = num_attention_heads * (2 if attn_output_gate else 1)
    return head_dim * (q_heads + 2 * num_kv_heads)


class TestColumnParallelLoraBPerRankDim(unittest.TestCase):
    """`_column_parallel_lora_b_per_rank_dim` for the qkv_proj path."""

    def test_non_qkv_path_falls_back_to_even_divide(self):
        pool = _make_pool(head_dim=128, num_attention_heads=32, num_kv_heads=8)
        # Any non-qkv module name must just divide evenly.
        self.assertEqual(
            pool._column_parallel_lora_b_per_rank_dim("gate_up_proj", 4096, 4),
            1024,
        )

    def test_qkv_no_replication_falls_back_to_even_divide(self):
        # tp_size <= num_kv_heads -> no replication, even split is correct.
        pool = _make_pool(head_dim=128, num_attention_heads=32, num_kv_heads=8)
        total = _qkv_total_output_dim(
            head_dim=128,
            num_attention_heads=32,
            num_kv_heads=8,
            attn_output_gate=False,
        )
        # total = 128 * (32 + 16) = 6144, /8 = 768 per rank.
        self.assertEqual(
            pool._column_parallel_lora_b_per_rank_dim("qkv_proj", total, 8),
            total // 8,
        )

    def test_qkv_with_kv_replication_qwen35_full_attn(self):
        # Qwen3.5-35B-A3B full-attention layer, TP=16:
        #   num_attention_heads=16, num_kv_heads=2, head_dim=256,
        #   attn_output_gate=True -> q_heads (gate-doubled) = 32.
        # Per-rank should be q(2*256=512) + 2*head_dim(2*256=512) = 1024,
        # NOT divide(9216, 16) = 576.
        pool = _make_pool(head_dim=256, num_attention_heads=16, num_kv_heads=2)
        total = _qkv_total_output_dim(
            head_dim=256,
            num_attention_heads=16,
            num_kv_heads=2,
            attn_output_gate=True,
        )
        self.assertEqual(total, 9216)
        self.assertEqual(
            pool._column_parallel_lora_b_per_rank_dim("qkv_proj", total, 16),
            1024,
        )

    def test_qkv_with_kv_replication_no_gate(self):
        # No attn_output_gate variant of the same scenario:
        # num_attention_heads=16, num_kv_heads=2, head_dim=256, tp=16.
        # Per-rank: q(1*256=256) + 2*head_dim(2*256=512) = 768,
        # NOT divide(5120, 16) = 320.
        pool = _make_pool(head_dim=256, num_attention_heads=16, num_kv_heads=2)
        total = _qkv_total_output_dim(
            head_dim=256,
            num_attention_heads=16,
            num_kv_heads=2,
            attn_output_gate=False,
        )
        self.assertEqual(total, 5120)
        self.assertEqual(
            pool._column_parallel_lora_b_per_rank_dim("qkv_proj", total, 16),
            768,
        )

    def test_qkv_replication_factor_8(self):
        # Replication factor (tp_size / num_kv_heads) is irrelevant: each
        # rank still owns exactly 1 KV head's worth of K and V. Verify by
        # bumping num_kv_heads to 4 (replication factor = 4) and checking
        # the per-rank dim is the same as for 8.
        for nkv in (1, 2, 4):
            pool = _make_pool(head_dim=128, num_attention_heads=32, num_kv_heads=nkv)
            total = _qkv_total_output_dim(
                head_dim=128,
                num_attention_heads=32,
                num_kv_heads=nkv,
                attn_output_gate=False,
            )
            # q_per_rank = (32 * 128) / 16 = 256; kv_per_rank_each = 128.
            # Result: 256 + 2*128 = 512.
            self.assertEqual(
                pool._column_parallel_lora_b_per_rank_dim("qkv_proj", total, 16),
                512,
                msg=f"failed for num_kv_heads={nkv}",
            )

    def test_qkv_missing_num_kv_heads_falls_back(self):
        # Defensive: if the text config lacks num_key_value_heads, the
        # helper must not crash and just return the naive even split.
        pool = LoRAMemoryPool.__new__(LoRAMemoryPool)
        pool.base_hf_config = types.SimpleNamespace(
            head_dim=128,
            hidden_size=4096,
            num_attention_heads=32,
        )
        self.assertEqual(
            pool._column_parallel_lora_b_per_rank_dim("qkv_proj", 6144, 8),
            768,
        )


def _make_fake_qkv_base_layer(
    *,
    head_dim: int,
    num_heads_total: int,
    num_kv_heads_total: int,
    tp_size: int,
):
    """Mirror the relevant attributes of `QKVParallelLinear` without
    constructing the real layer (which requires a CUDA device and a
    distributed group)."""
    if tp_size >= num_kv_heads_total:
        num_kv_heads_per_rank = 1
        num_kv_head_replicas = tp_size // num_kv_heads_total
    else:
        num_kv_heads_per_rank = num_kv_heads_total // tp_size
        num_kv_head_replicas = 1

    num_heads_per_rank = num_heads_total // tp_size
    q_proj_shard_size = num_heads_per_rank * head_dim
    kv_proj_shard_size = num_kv_heads_per_rank * head_dim

    return types.SimpleNamespace(
        q_proj_shard_size=q_proj_shard_size,
        kv_proj_shard_size=kv_proj_shard_size,
        num_kv_head_replicas=num_kv_head_replicas,
        output_sizes=[
            num_heads_per_rank * head_dim * tp_size,
            num_kv_heads_per_rank * head_dim * tp_size,
            num_kv_heads_per_rank * head_dim * tp_size,
        ],
    )


def _make_qkv_lora_layer(base_layer):
    """Build a `QKVParallelLinearWithLoRA` skipping `__init__` (which
    expects a real `QKVParallelLinear` with parameters on a CUDA device).
    Only `slice_lora_b_weights` is exercised, and it only reads
    `self.base_layer`."""
    layer = QKVParallelLinearWithLoRA.__new__(QKVParallelLinearWithLoRA)
    layer.base_layer = base_layer
    return layer


class TestQKVSliceLoraBWithReplication(unittest.TestCase):
    """`QKVParallelLinearWithLoRA.slice_lora_b_weights` with
    ``num_kv_head_replicas > 1``."""

    def _make_unreplicated_b(
        self,
        *,
        head_dim: int,
        num_heads_total: int,
        num_kv_heads_total: int,
        rank: int,
        attn_output_gate: bool,
    ) -> torch.Tensor:
        """Build a PEFT-format LoRA B tensor, layout
        ``[q_total, k_total, v_total]`` along dim 0.

        Each row is filled with a deterministic value encoding (section,
        head, head-row) so per-rank slices can be verified positionally:
            q rows: 1.0 + head_idx * 100 + within_head_row
            k rows: 2.0 + head_idx * 100 + within_head_row
            v rows: 3.0 + head_idx * 100 + within_head_row
        """
        q_heads = num_heads_total * (2 if attn_output_gate else 1)
        q_total = head_dim * q_heads
        k_total = head_dim * num_kv_heads_total
        v_total = head_dim * num_kv_heads_total

        def _fill(section_id: int, num_heads: int) -> torch.Tensor:
            block = torch.empty(num_heads * head_dim, rank, dtype=torch.float32)
            for h in range(num_heads):
                for r in range(head_dim):
                    block[h * head_dim + r, :] = section_id + h * 100 + r * 0.001
            return block

        q_block = _fill(1, q_heads)
        k_block = _fill(2, num_kv_heads_total)
        v_block = _fill(3, num_kv_heads_total)
        return torch.cat([q_block, k_block, v_block], dim=0)

    def test_qwen35_full_attn_tp16_per_rank_shape(self):
        # Shape end-to-end check: with TP=16, the slice must produce
        # 1024 rows (= 512 q + 256 k + 256 v) — matching the buffer that
        # `_column_parallel_lora_b_per_rank_dim` allocates for this case.
        head_dim = 256
        num_heads_total = 32  # gate-doubled q heads passed to QKVParallelLinear
        num_kv_heads_total = 2
        tp_size = 16
        rank = 16

        base = _make_fake_qkv_base_layer(
            head_dim=head_dim,
            num_heads_total=num_heads_total,
            num_kv_heads_total=num_kv_heads_total,
            tp_size=tp_size,
        )
        layer = _make_qkv_lora_layer(base)

        # Build B as the model actually trained it: q has gate-doubled
        # heads (`attn_output_gate=True`), K/V are at 2 heads.
        # The `attn_output_gate=True` here mirrors how Qwen3.5's q_proj
        # is trained as a single `Linear` with 2*num_heads*head_dim
        # outputs in PEFT.
        B = self._make_unreplicated_b(
            head_dim=head_dim,
            num_heads_total=num_heads_total // 2,  # base num_attention_heads
            num_kv_heads_total=num_kv_heads_total,
            rank=rank,
            attn_output_gate=True,
        )
        self.assertEqual(B.shape, (9216, rank))

        for tp_rank in range(tp_size):
            sliced = layer.slice_lora_b_weights(B, tp_rank)
            self.assertEqual(
                sliced.shape,
                (1024, rank),
                msg=f"tp_rank={tp_rank}",
            )

    def test_qwen35_full_attn_tp16_kv_replicas_share_kv_slice(self):
        # Ranks that share a KV head (because of replication) must see
        # the *same* K and V rows, while their Q rows differ. With TP=16,
        # num_kv_heads=2, num_kv_head_replicas=8: ranks 0..7 share KV
        # head 0; ranks 8..15 share KV head 1.
        head_dim = 256
        num_heads_total = 32
        num_kv_heads_total = 2
        tp_size = 16
        rank = 4

        base = _make_fake_qkv_base_layer(
            head_dim=head_dim,
            num_heads_total=num_heads_total,
            num_kv_heads_total=num_kv_heads_total,
            tp_size=tp_size,
        )
        layer = _make_qkv_lora_layer(base)

        B = self._make_unreplicated_b(
            head_dim=head_dim,
            num_heads_total=num_heads_total // 2,
            num_kv_heads_total=num_kv_heads_total,
            rank=rank,
            attn_output_gate=True,
        )

        slices = [layer.slice_lora_b_weights(B, r) for r in range(tp_size)]

        q_size_per_rank = base.q_proj_shard_size  # 512
        kv_size_per_rank = base.kv_proj_shard_size  # 256

        # Ranks 0..7 -> KV head 0; ranks 8..15 -> KV head 1.
        for r in range(tp_size):
            kv_block_start = q_size_per_rank
            k_slice = slices[r][kv_block_start : kv_block_start + kv_size_per_rank]
            v_slice = slices[r][kv_block_start + kv_size_per_rank :]
            # K block of B starts with section_id=2.0; the KV head this
            # rank belongs to is r // 8.
            expected_kv_head = r // 8
            self.assertAlmostEqual(
                float(k_slice[0, 0]), 2.0 + expected_kv_head * 100, places=4
            )
            self.assertAlmostEqual(
                float(v_slice[0, 0]), 3.0 + expected_kv_head * 100, places=4
            )

        # Within a replication group, K and V tensors must match exactly.
        for r in range(1, 8):
            self.assertTrue(
                torch.equal(
                    slices[0][q_size_per_rank:],
                    slices[r][q_size_per_rank:],
                ),
                msg=f"rank 0 vs {r}",
            )
        for r in range(9, 16):
            self.assertTrue(
                torch.equal(
                    slices[8][q_size_per_rank:],
                    slices[r][q_size_per_rank:],
                ),
                msg=f"rank 8 vs {r}",
            )

    def test_no_replication_unchanged_behavior(self):
        # Sanity: with `tp_size <= num_kv_heads`, no replication happens
        # (`num_kv_head_replicas == 1`), and the slice should match a
        # straightforward per-rank cut of each of q, k, v.
        head_dim = 128
        num_heads_total = 32
        num_kv_heads_total = 8
        tp_size = 4
        rank = 4

        base = _make_fake_qkv_base_layer(
            head_dim=head_dim,
            num_heads_total=num_heads_total,
            num_kv_heads_total=num_kv_heads_total,
            tp_size=tp_size,
        )
        self.assertEqual(base.num_kv_head_replicas, 1)
        layer = _make_qkv_lora_layer(base)

        B = self._make_unreplicated_b(
            head_dim=head_dim,
            num_heads_total=num_heads_total,
            num_kv_heads_total=num_kv_heads_total,
            rank=rank,
            attn_output_gate=False,
        )

        for tp_rank in range(tp_size):
            sliced = layer.slice_lora_b_weights(B, tp_rank)
            # Per-rank: q = 8 heads * 128 = 1024; k = 2 heads * 128 = 256;
            # v = 256 -> total 1536.
            self.assertEqual(sliced.shape, (1536, rank))


if __name__ == "__main__":
    unittest.main()
