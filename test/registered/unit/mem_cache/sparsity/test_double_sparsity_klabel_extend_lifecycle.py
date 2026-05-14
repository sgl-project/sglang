"""K_label extend-lifecycle gate.

Plan note (M2 gate): the dense fast-path-first branch in `RadixAttention.forward`
must NOT prevent K_label writes during prefill. The two-gate model splits
"DS enabled" (which always writes K_label) from "DS selecting this step"
(decode-only). This test pins that contract by exercising
`construct_representations` (extend) and `update_representations` (decode)
directly with a mocked forward_batch, asserting K_label receives prefill
writes and decode writes alike, and stays untouched in extend-with-no-extend
modes. CPU-only — uses the torch reference path.
"""

import unittest
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import MagicMock

import torch

from sglang.srt.mem_cache.sparsity.algorithms.double_sparsity import (
    DoubleSparsityAlgorithm,
)
from sglang.srt.mem_cache.sparsity.algorithms.double_sparsity_config import (
    DoubleSparsityRuntimeConfig,
    parse_calibration_file,
)
from sglang.srt.mem_cache.sparsity.core.sparse_coordinator import SparseConfig
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="stage-a-test-cpu")


FIXTURE_PATH = Path(__file__).parent / "_fixtures" / "tiny_ds_calibration.json"


class _Mode:
    """Minimal shim for ForwardMode."""

    def __init__(self, name: str):
        self.name = name

    def is_extend(self) -> bool:
        return self.name == "extend"

    def is_decode_or_idle(self) -> bool:
        return self.name in ("decode", "idle")


@dataclass
class _FakeBatch:
    forward_mode: _Mode
    out_cache_loc: torch.Tensor
    save_kv_cache: bool = True


def _build_algo(num_kv_heads_local: int = 4, num_layers: int = 4, head_dim: int = 16):
    calib = parse_calibration_file(FIXTURE_PATH)
    rt = DoubleSparsityRuntimeConfig(
        heavy_channels=8,
        token_budget=16,
        recent_tokens=2,
        sink_tokens=1,
        min_seq_len=8,
        max_selected_per_request=64,
        gqa_reduction="max_abs",
        klabel_dtype="bf16",
    )
    sc = SparseConfig(algorithm="double_sparsity", backend="fa3", page_size=1)
    algo = DoubleSparsityAlgorithm(
        sc,
        torch.device("cpu"),
        runtime_config=rt,
        calibration=calib,
        tp_size=1,
        tp_rank=0,
        num_kv_heads_local=num_kv_heads_local,
        num_q_heads_local=num_kv_heads_local * 2,
        head_dim=head_dim,
    )
    return algo


def _mock_pools(num_tokens: int, num_kv_heads: int, head_dim: int, num_layers: int):
    k_buffers = [
        torch.zeros(num_tokens, num_kv_heads, head_dim, dtype=torch.bfloat16)
        for _ in range(num_layers)
    ]
    token_pool = MagicMock()
    token_pool.get_key_buffer.side_effect = lambda layer_id: k_buffers[layer_id]
    req_pool = MagicMock()
    # Commit 0: DoubleSparsityAlgorithm.initialize_representation_pool now reads
    # req_to_token_pool.req_to_token.shape[1] to size the selection scratch.
    # Provide a real tensor so the K_label-extend tests still construct.
    req_pool.req_to_token = torch.zeros(1, num_tokens, dtype=torch.int32)
    return token_pool, req_pool, k_buffers


class TestExtendLifecycle(CustomTestCase):
    NUM_TOKENS = 32
    H_KV = 4
    D = 16
    NUM_LAYERS = 4

    def setUp(self):
        self.algo = _build_algo(self.H_KV, self.NUM_LAYERS, self.D)
        self.token_pool, self.req_pool, self.k_buffers = _mock_pools(
            self.NUM_TOKENS, self.H_KV, self.D, self.NUM_LAYERS
        )
        self.algo.initialize_representation_pool(
            start_layer=0,
            end_layer=self.NUM_LAYERS,
            token_to_kv_pool=self.token_pool,
            req_to_token_pool=self.req_pool,
            states=MagicMock(),
        )

    def test_k_label_allocated_per_layer(self):
        for layer_id in range(self.NUM_LAYERS):
            self.assertIn(layer_id, self.algo.k_label)
            kl = self.algo.k_label[layer_id]
            self.assertEqual(
                kl.shape,
                (self.NUM_TOKENS, self.H_KV, self.algo.runtime_config.heavy_channels),
            )
            self.assertEqual(kl.dtype, torch.bfloat16)
            # Allocated empty/zero
            self.assertTrue(torch.equal(kl, torch.zeros_like(kl)))

    def test_construct_writes_k_label_during_extend(self):
        # Simulate a prefill chunk: 6 new tokens written to physical ids [10..16).
        N = 6
        out_loc = torch.tensor([10, 11, 12, 13, 14, 15], dtype=torch.int64)
        torch.manual_seed(7)
        k_new = torch.randn(N, self.H_KV, self.D, dtype=torch.bfloat16)
        # Place the new K rows into the pool's K buffer at out_loc — this is
        # what FA3's set_kv_buffer does just before SparseCoordinator.attention_end.
        self.k_buffers[1][out_loc] = k_new

        fb = _FakeBatch(_Mode("extend"), out_loc)
        self.algo.construct_representations(
            layer_id=1,
            req_pool_indices=torch.tensor([0]),
            seq_lens=torch.tensor([6]),
            k_buffer=self.k_buffers[1],
            forward_batch=fb,
        )

        kl = self.algo.k_label[1]
        for i, loc in enumerate(out_loc.tolist()):
            for h in range(self.H_KV):
                expected = k_new[i, h, self.algo.channel_indices[1][h].long()]
                self.assertTrue(torch.equal(kl[loc, h].float(), expected.float()))

        # Untouched rows must remain zero
        untouched = [t for t in range(self.NUM_TOKENS) if t not in out_loc.tolist()]
        for t in untouched:
            self.assertTrue(torch.equal(kl[t], torch.zeros_like(kl[t])))

    def test_update_writes_k_label_during_decode(self):
        # Simulate one decode step for batch=2.
        N = 2
        out_loc = torch.tensor([20, 21], dtype=torch.int64)
        torch.manual_seed(8)
        k_new = torch.randn(N, self.H_KV, self.D, dtype=torch.bfloat16)
        self.k_buffers[2][out_loc] = k_new

        fb = _FakeBatch(_Mode("decode"), out_loc)
        self.algo.update_representations(
            layer_id=2,
            req_pool_indices=torch.tensor([0, 1]),
            seq_lens=torch.tensor([100, 105]),
            k_buffer=self.k_buffers[2],
            forward_batch=fb,
        )

        kl = self.algo.k_label[2]
        for i, loc in enumerate(out_loc.tolist()):
            for h in range(self.H_KV):
                expected = k_new[i, h, self.algo.channel_indices[2][h].long()]
                self.assertTrue(torch.equal(kl[loc, h].float(), expected.float()))

    def test_construct_in_decode_mode_is_noop(self):
        # construct_representations (extend hook) must not fire in decode mode.
        fb = _FakeBatch(_Mode("decode"), torch.tensor([0], dtype=torch.int64))
        before = self.algo.k_label[0].clone()
        self.algo.construct_representations(
            layer_id=0,
            req_pool_indices=torch.tensor([0]),
            seq_lens=torch.tensor([1]),
            k_buffer=self.k_buffers[0],
            forward_batch=fb,
        )
        self.assertTrue(torch.equal(self.algo.k_label[0], before))

    def test_update_in_extend_mode_is_noop(self):
        # update_representations (decode hook) must not fire in extend mode.
        fb = _FakeBatch(_Mode("extend"), torch.tensor([0], dtype=torch.int64))
        before = self.algo.k_label[0].clone()
        self.algo.update_representations(
            layer_id=0,
            req_pool_indices=torch.tensor([0]),
            seq_lens=torch.tensor([1]),
            k_buffer=self.k_buffers[0],
            forward_batch=fb,
        )
        self.assertTrue(torch.equal(self.algo.k_label[0], before))


class TestSaveKvCacheFalse(CustomTestCase):
    """save_kv_cache=False must leave K_label untouched (otherwise side cache desyncs)."""

    def setUp(self):
        self.algo = _build_algo()
        self.token_pool, self.req_pool, self.k_buffers = _mock_pools(16, 4, 16, 4)
        self.algo.initialize_representation_pool(
            start_layer=0,
            end_layer=4,
            token_to_kv_pool=self.token_pool,
            req_to_token_pool=self.req_pool,
            states=MagicMock(),
        )

    def test_extend_with_save_kv_cache_false_does_not_write(self):
        N = 3
        out_loc = torch.tensor([1, 2, 3], dtype=torch.int64)
        torch.manual_seed(9)
        k_new = torch.randn(N, 4, 16, dtype=torch.bfloat16)
        self.k_buffers[0][out_loc] = k_new

        fb = _FakeBatch(_Mode("extend"), out_loc, save_kv_cache=False)
        before = self.algo.k_label[0].clone()
        self.algo.construct_representations(
            layer_id=0,
            req_pool_indices=torch.tensor([0]),
            seq_lens=torch.tensor([3]),
            k_buffer=self.k_buffers[0],
            forward_batch=fb,
        )
        self.assertTrue(torch.equal(self.algo.k_label[0], before))

    def test_decode_with_save_kv_cache_false_does_not_write(self):
        out_loc = torch.tensor([4], dtype=torch.int64)
        torch.manual_seed(10)
        k_new = torch.randn(1, 4, 16, dtype=torch.bfloat16)
        self.k_buffers[0][out_loc] = k_new

        fb = _FakeBatch(_Mode("decode"), out_loc, save_kv_cache=False)
        before = self.algo.k_label[0].clone()
        self.algo.update_representations(
            layer_id=0,
            req_pool_indices=torch.tensor([0]),
            seq_lens=torch.tensor([100]),
            k_buffer=self.k_buffers[0],
            forward_batch=fb,
        )
        self.assertTrue(torch.equal(self.algo.k_label[0], before))


class TestRadixAttentionSaveKvCacheRealPath(CustomTestCase):
    """Real-path integration test: RadixAttention.forward(..., save_kv_cache=False)
    must skip coordinator.attention_end entirely.

    The existing TestSaveKvCacheFalse pins the algorithm's defensive
    `getattr(forward_batch, 'save_kv_cache', True)` check — useful but
    the real call site never sets that attribute, so that test misses
    the actual code path. This test pins the RadixAttention.forward
    integration where `save_kv_cache` is a function argument.
    """

    def _build_attn(self):
        from sglang.srt.layers.radix_attention import RadixAttention

        attn = RadixAttention(
            num_heads=4, head_dim=16, scaling=0.25, num_kv_heads=2, layer_id=0
        )
        attn.ds_enabled = True
        return attn

    def _build_fb(self, mode: str):
        # Minimal forward_batch + attn_backend for _forward_inner's "else" path.
        attn_backend = MagicMock()
        attn_backend.forward = MagicMock(return_value=torch.zeros(1, 4 * 16))
        attn_backend.forward_metadata = None

        class _M:
            def __init__(self, name):
                self.name = name

            def is_extend(self):
                return self.name == "extend"

            def is_decode_or_idle(self):
                return self.name in ("decode", "idle")

        fb = MagicMock()
        fb.forward_mode = _M(mode)
        fb.attn_backend = attn_backend
        return fb

    def test_save_kv_cache_false_skips_attention_end_decode(self):
        from unittest.mock import patch

        attn = self._build_attn()
        fb = self._build_fb("decode")
        coord = MagicMock()
        # Make get_sparse_coordinator return our stubbed coordinator.
        with patch(
            "sglang.srt.mem_cache.sparsity.get_sparse_coordinator", return_value=coord
        ):
            q = torch.zeros(1, 4 * 16)
            k = torch.zeros(1, 2 * 16)
            v = torch.zeros(1, 2 * 16)
            attn.forward(q, k, v, fb, save_kv_cache=False)

        coord.attention_end.assert_not_called()
        # attention_begin still fires (it rewrites FA3 metadata before
        # the dense kernel runs; not bound by save_kv_cache).
        coord.attention_begin.assert_called_once()

    def test_save_kv_cache_true_calls_attention_end(self):
        from unittest.mock import patch

        attn = self._build_attn()
        fb = self._build_fb("decode")
        coord = MagicMock()
        with patch(
            "sglang.srt.mem_cache.sparsity.get_sparse_coordinator", return_value=coord
        ):
            q = torch.zeros(1, 4 * 16)
            k = torch.zeros(1, 2 * 16)
            v = torch.zeros(1, 2 * 16)
            attn.forward(q, k, v, fb, save_kv_cache=True)

        coord.attention_end.assert_called_once()


if __name__ == "__main__":
    unittest.main()
