"""Unit tests for build_hot_token_lm_head (hot-token draft lm_head assembly).

The TP>1 path is exercised with real collectives over a CPU gloo group
(4 processes), mirroring a vocab-sharded ParallelLMHead without loading a
model: each rank owns a shard (the last one ragged, covering padding), and
the assembled head must equal full_weight[hot_ids] exactly on every rank.
"""

import unittest
from types import SimpleNamespace

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from sglang.srt.layers.vocab_parallel_embedding import (
    VocabParallelEmbeddingShardIndices,
    vocab_range_from_global_vocab_size,
)
from sglang.srt.speculative.eagle_worker_common import build_hot_token_lm_head
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

_V, _H, _TP, _NHOT = 1000, 32, 4, 200


class _GlooGroup:
    """The subset of GroupCoordinator the builder needs, over gloo."""

    def __init__(self, group, size):
        self._group = group
        self.size = size

    def all_gather_into_tensor(self, output, input):
        chunks = [torch.empty_like(input) for _ in range(self.size)]
        dist.all_gather(chunks, input.contiguous(), group=self._group)
        output.copy_(torch.stack(chunks).reshape(output.shape))


def _worker(rank, world_size, port):
    dist.init_process_group(
        "gloo", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=world_size
    )
    try:
        torch.manual_seed(0)  # identical full weight on every rank
        full = torch.randn(_V, _H, dtype=torch.float32)
        hot = torch.linspace(0, _V - 1, _NHOT).long()
        expected = full[hot]

        # Mirror VocabParallelEmbedding._get_indices: org-only vocab of 1000
        # padded to 1024 across 4 ranks (256 each); the last shard is ragged
        # at 1000 and carries the org padding.
        pstart, pend = vocab_range_from_global_vocab_size(1024, rank, world_size)
        si = VocabParallelEmbeddingShardIndices(
            pstart,
            pend,
            _V,
            _V,  # padded added (empty)
            min(pstart, _V),
            min(pend, _V),  # org (clamped)
            _V,
            _V,  # added (empty)
        )
        module = SimpleNamespace(
            tp_size=world_size,
            tp_group=_GlooGroup(dist.group.WORLD, world_size),
            shard_indices=si,
            embedding_dim=_H,
            num_embeddings=_V,
        )
        shard = full[min(pstart, _V) : min(pend, _V)]

        out = build_hot_token_lm_head(
            head_weight=shard,
            lm_head_module=module,
            hot_token_id=hot,
            vocab_size=_V,
        )
        assert out.shape == (_NHOT, _H), (rank, out.shape)
        assert out.dtype == torch.float32, (rank, out.dtype)
        assert torch.equal(out, expected), f"rank {rank} assembled wrong rows"
    finally:
        dist.destroy_process_group()


class TestBuildHotTokenLmHead(CustomTestCase):
    def test_tp4_assembly_is_exact_on_every_rank(self):
        mp.spawn(
            _worker,
            args=(_TP, _get_free_port()),
            nprocs=_TP,
            join=True,
        )

    def test_tp1_fast_path_selects_rows_and_wraps_parameters(self):
        full = torch.randn(128, 8)
        hot = torch.tensor([0, 3, 7, 120], dtype=torch.int64)
        # A plain module-less weight: the historical TP=1 semantics.
        out = build_hot_token_lm_head(
            head_weight=full,
            lm_head_module=None,
            hot_token_id=hot,
            vocab_size=128,
        )
        self.assertTrue(torch.equal(out, full[hot]))
        self.assertNotIsInstance(out, torch.nn.Parameter)
        # A Parameter stays a Parameter after reduction.
        out_p = build_hot_token_lm_head(
            head_weight=torch.nn.Parameter(full),
            lm_head_module=None,
            hot_token_id=hot,
            vocab_size=128,
        )
        self.assertIsInstance(out_p, torch.nn.Parameter)
        self.assertTrue(torch.equal(out_p.data, full[hot]))

    def test_rejects_empty_and_out_of_range_maps(self):
        full = torch.randn(128, 8)
        with self.assertRaisesRegex(ValueError, "non-empty 1-D"):
            build_hot_token_lm_head(
                head_weight=full,
                lm_head_module=None,
                hot_token_id=torch.empty(0, dtype=torch.int64),
                vocab_size=128,
            )
        with self.assertRaisesRegex(ValueError, r"outside \[0, 128\)"):
            build_hot_token_lm_head(
                head_weight=full,
                lm_head_module=None,
                hot_token_id=torch.tensor([5, 128], dtype=torch.int64),
                vocab_size=128,
            )

    def test_rejects_packed_quantized_rows_loudly(self):
        # NVFP4-style packed rows: uint8 [vocab, hidden/2] plus per-block
        # scales that cannot follow selected rows. Must raise, not bind
        # garbage (the historical behavior at TP=1).
        packed = torch.randint(0, 255, (128, 4), dtype=torch.uint8)
        module = SimpleNamespace(
            tp_size=1,
            tp_group=None,
            embedding_dim=8,
            num_embeddings=128,
        )
        with self.assertRaisesRegex(ValueError, "extractable rows"):
            build_hot_token_lm_head(
                head_weight=packed,
                lm_head_module=module,
                hot_token_id=torch.tensor([0, 1], dtype=torch.int64),
                vocab_size=128,
            )


def _get_free_port():
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


if __name__ == "__main__":
    unittest.main()
