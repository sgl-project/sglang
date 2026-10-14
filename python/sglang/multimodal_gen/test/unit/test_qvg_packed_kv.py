# SPDX-License-Identifier: Apache-2.0
"""Unit tests for Quant-VideoGen packed KV-cache (QVGPackedCausalKVCache).

- config parsing + defaults: CPU, no deps.
- packed-storage equivalence vs the dense cache (quant disabled -> bf16
  segments): CPU-only, exercises the segment / sliding-window / eviction /
  reconstruction logic bit-exactly without needing quant-videogen or a GPU.
- quant memory saving + reconstruction: requires CUDA + quant-videogen.
"""

import importlib.util
import unittest
from types import SimpleNamespace

import torch

from sglang.multimodal_gen.configs.quantization.qvg_kv import QVGKVQuantArgs
from sglang.multimodal_gen.runtime.layers.kvcache.causal_attention_cache import (
    CausalSelfAttentionKVCache,
)
from sglang.multimodal_gen.runtime.layers.kvcache.qvg_packed_cache import (
    QVGPackedCausalKVCache,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.causal_denoising import (
    CausalDMDDenoisingStage,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.lingbot_world.lingbot_world_causal_denoising import (
    LingBotWorldCausalDMDDenoisingStage,
)

_HAS_QVG = importlib.util.find_spec("quant_videogen") is not None
_HAS_CUDA = torch.cuda.is_available()


def _base(B, W, H, D, sink, dev):
    return CausalSelfAttentionKVCache(
        k=torch.zeros(B, W, H, D, device=dev),
        v=torch.zeros(B, W, H, D, device=dev),
        global_end_index=torch.zeros(1, dtype=torch.long, device=dev),
        local_end_index=torch.zeros(1, dtype=torch.long, device=dev),
        cache_size=W,
        sink_tokens=sink,
        attention_window_size=W,
    )


def _packed(B, W, H, D, sink, dev, quant, use_int_indices=False):
    return QVGPackedCausalKVCache(
        batch_size=B,
        cache_size=W,
        num_heads=H,
        head_dim=D,
        dtype=torch.float32,
        device=torch.device(dev),
        use_int_indices=use_int_indices,
        global_end_index=torch.zeros(1, dtype=torch.long, device=dev),
        local_end_index=torch.zeros(1, dtype=torch.long, device=dev),
        sink_tokens=sink,
        attention_window_size=W,
        quant_args=quant,
    )


def _replay(cache, chunks, C, H, D, B, nsteps, rwt, data, dev):
    views = []
    for ci in range(chunks):
        g = ci * C
        for step in range(nsteps):
            k, v = data[(ci, step)]
            view = cache.update_and_get_attention_kv(
                key=k, value=v, current_chunk_start=g, recent_window_tokens=rwt
            )
        views.append((view.k.clone(), view.v.clone()))
    return views


class TestQVGKVQuantArgs(unittest.TestCase):
    def test_off_by_default(self):
        self.assertFalse(QVGKVQuantArgs().enabled)
        self.assertFalse(QVGKVQuantArgs.from_dict({}).enabled)

    def test_master_flag_defaults(self):
        a = QVGKVQuantArgs.from_dict({"kv_cache_quant": "int4"})
        self.assertTrue(a.enabled)
        self.assertEqual(
            (
                a.bits,
                a.stages,
                a.centroids,
                a.block_size,
                a.kmeans_iters,
                a.asymmetric,
                a.keep_recent_chunks,
                a.sink,
                a.sink_keep_chunks,
            ),
            (4, 1, 128, 64, 2, False, 1, True, 0),
        )

    def test_overrides_and_validation(self):
        a = QVGKVQuantArgs.from_dict(
            {
                "kv_cache_quant": "int2",
                "kv_cache_quant_stages": 3,
                "kv_cache_quant_sink": 0,
            }
        )
        self.assertEqual((a.bits, a.stages, a.sink), (2, 3, False))
        with self.assertRaises(ValueError):
            QVGKVQuantArgs.from_dict({"kv_cache_quant": "int8"})


class TestPackedStorageEquivalence(unittest.TestCase):
    """quant disabled -> packed cache stores bf16 segments; must match the
    dense cache bit-for-bit across the sliding-window + eviction lifecycle."""

    def _run(self, rwt):
        B, H, D, C, sink, chunks, nsteps = 1, 2, 4, 8, 8, 8, 3
        W = C * 5
        dev = "cpu"
        torch.manual_seed(0)
        data = {
            (ci, s): (torch.randn(B, C, H, D), torch.randn(B, C, H, D))
            for ci in range(chunks)
            for s in range(nsteps)
        }
        base = _base(B, W, H, D, sink, dev)
        packed = _packed(B, W, H, D, sink, dev, QVGKVQuantArgs())  # disabled
        ob = _replay(base, chunks, C, H, D, B, nsteps, rwt, data, dev)
        op = _replay(packed, chunks, C, H, D, B, nsteps, rwt, data, dev)
        for i, ((bk, bv), (pk, pv)) in enumerate(zip(ob, op)):
            self.assertEqual(bk.shape, pk.shape, f"chunk {i} shape")
            self.assertTrue(torch.equal(bk, pk), f"chunk {i} K mismatch")
            self.assertTrue(torch.equal(bv, pv), f"chunk {i} V mismatch")

    def test_full_window(self):
        self._run(rwt=None)

    def test_recent_window_selection(self):
        self._run(rwt=16)

    def test_host_index_cursors_reset_without_device_updates(self):
        cache = _packed(
            B=1,
            W=16,
            H=2,
            D=4,
            sink=0,
            dev="cpu",
            quant=QVGKVQuantArgs(),
            use_int_indices=True,
        )
        cache.update_and_get_attention_kv(
            key=torch.ones(1, 4, 2, 4),
            value=torch.ones(1, 4, 2, 4),
            current_chunk_start=0,
        )
        self.assertEqual(cache.global_end_index_int, 4)
        self.assertEqual(cache.local_end_index_int, 4)
        self.assertEqual(int(cache.global_end_index.item()), 0)
        self.assertEqual(int(cache.local_end_index.item()), 0)

        cache.reset_indices()
        self.assertEqual(cache.global_end_index_int, 0)
        self.assertEqual(cache.local_end_index_int, 0)

    def test_quantization_is_rejected_by_unsupported_causal_stages(self):
        stage = CausalDMDDenoisingStage.__new__(CausalDMDDenoisingStage)
        stage.sink_size = 0
        stage.sliding_window_num_frames = 1

        with self.assertRaisesRegex(ValueError, "does not support QVG"):
            stage._apply_causal_cache_overrides(
                SimpleNamespace(),
                SimpleNamespace(
                    pipeline_config=SimpleNamespace(),
                    kv_cache_quant_config=QVGKVQuantArgs(bits=4),
                ),
            )

    def test_quantization_is_available_for_lingbot(self):
        stage = LingBotWorldCausalDMDDenoisingStage.__new__(
            LingBotWorldCausalDMDDenoisingStage
        )
        stage._kv_quant_args = QVGKVQuantArgs(bits=4)
        stage.num_transformer_blocks = 1

        caches = stage._allocate_causal_kv_cache(
            batch_size=1,
            kv_cache_size=8,
            num_attention_heads=1,
            attention_head_dim=4,
            dtype=torch.float32,
            device=torch.device("cpu"),
        )

        self.assertIsInstance(caches[0], QVGPackedCausalKVCache)


@unittest.skipUnless(
    _HAS_CUDA and _HAS_QVG,
    "needs CUDA + quant-videogen",
)
class TestPackedQuantMemory(unittest.TestCase):
    def test_saves_memory_and_reconstructs(self):
        B, H, D, C, sink, chunks, nsteps = 1, 8, 128, 512, 512, 12, 2
        W = C * 6
        dev = "cuda"
        q = QVGKVQuantArgs.from_dict({"kv_cache_quant": "int4"})
        torch.manual_seed(1)
        data = {
            (ci, s): (
                torch.randn(B, C, H, D, device=dev),
                torch.randn(B, C, H, D, device=dev),
            )
            for ci in range(chunks)
            for s in range(nsteps)
        }
        base = _base(B, W, H, D, sink, dev)
        packed = _packed(B, W, H, D, sink, dev, q)
        ob = _replay(base, chunks, C, H, D, B, nsteps, None, data, dev)
        op = _replay(packed, chunks, C, H, D, B, nsteps, None, data, dev)
        dense_bytes = base.k.numel() * base.k.element_size() * 2
        self.assertLess(
            packed.resident_nbytes(),
            dense_bytes * 0.7,
            "packed cache should save >30% resident memory",
        )
        bk, _ = ob[-1]
        pk, _ = op[-1]
        self.assertEqual(bk.shape, pk.shape)
        mse = ((bk - pk) ** 2).mean().item()
        self.assertLess(mse, 1.0, "int4 reconstruction error too large")


if __name__ == "__main__":
    unittest.main()
