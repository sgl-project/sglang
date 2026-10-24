"""Decode KV must never be written through req_to_token entries past
``owned_kv_len()``, nor after the request's row is released."""

from __future__ import annotations

import importlib.util
import unittest
from types import SimpleNamespace

from sglang.test.ci.ci_register import register_mlx_ci
from sglang.test.test_utils import CustomTestCase

register_mlx_ci(est_time=5, suite="stage-a-unit-test-mlx")

_HAS_MLX = (
    importlib.util.find_spec("mlx") is not None
    and importlib.util.find_spec("mlx_lm") is not None
)

if _HAS_MLX:
    import mlx.core as mx
    import torch
    from mlx_lm.models import llama

    from sglang.srt.hardware_backend.mlx.aot import MlxAOTKernelSet
    from sglang.srt.hardware_backend.mlx.kv_cache import (
        find_attention_layers,
        get_layer_window_sizes,
        patch_model_attention,
    )
    from sglang.srt.hardware_backend.mlx.kv_cache.layout import MlxModelCacheLayout
    from sglang.srt.hardware_backend.mlx.model_runner import MlxModelRunner
    from sglang.srt.hardware_backend.mlx.tp_worker import MlxTpModelWorker
    from sglang.srt.mem_cache.memory_pool import ReqToTokenPool

PROMPT_LEN = 5
DECODE_STEPS = 5
OWN_SLOTS = [1, 2, 3, 4, 5, 6]  # prompt slots + the one allocated decode slot
FOREIGN_SLOTS = [20, 21, 22, 23]  # stale row entries left by a previous owner


def _runner(req_to_token_pool):
    model = llama.Model(
        llama.ModelArgs(
            model_type="llama",
            hidden_size=32,
            num_hidden_layers=2,
            intermediate_size=64,
            num_attention_heads=2,
            rms_norm_eps=1e-5,
            vocab_size=64,
            head_dim=16,
            num_key_value_heads=1,
        )
    )
    patch_model_attention(model)
    layers, attrs = find_attention_layers(model)
    runner = MlxModelRunner.__new__(MlxModelRunner)
    runner.model = model
    runner.disable_radix_cache = False
    runner._cache_layout = MlxModelCacheLayout.from_attention_discovery(
        layers, attrs, layer_window_sizes=get_layer_window_sizes(model)
    )
    runner._max_seq_len = 64
    runner._cache_pool = []
    runner._req_caches = {}
    runner._req_token_ids = {}
    runner._req_sampling = {}
    runner._req_pool_idx = {}
    runner._req_synced_offset = {}
    runner._attention_kv_pool = None
    runner._decode_step_ct = 0
    runner._clear_steps = 0
    runner._aot_kernels = MlxAOTKernelSet()
    runner._pool_size = 32
    runner.init_cache_pools(req_to_token_pool)
    return runner


@unittest.skipUnless(_HAS_MLX, "requires mlx + mlx_lm")
class TestDecodeKvRelease(CustomTestCase):
    def _decoded_runner(self):
        req_to_token_pool = ReqToTokenPool(
            size=2, max_context_len=64, device="cpu", enable_memory_saver=False
        )
        row = req_to_token_pool.req_to_token[0]
        row[: len(OWN_SLOTS)] = torch.tensor(OWN_SLOTS)
        row[len(OWN_SLOTS) : len(OWN_SLOTS) + len(FOREIGN_SLOTS)] = torch.tensor(
            FOREIGN_SLOTS
        )
        runner = _runner(req_to_token_pool)
        prompt = [3, 14, 15, 9, 2]
        runner.prefill(
            req_id="r",
            new_token_ids=prompt,
            full_token_ids=prompt,
            prefix_slot_ids=[],
            new_slot_ids=OWN_SLOTS[:PROMPT_LEN],
            req_pool_idx=0,
        )
        for _ in range(DECODE_STEPS):
            runner.decode_batch(["r"])
        return runner

    def _assert_foreign_slots_untouched(self, runner):
        pool = runner._attention_kv_pool
        foreign_k, foreign_v = pool.get_kv(0, mx.array(FOREIGN_SLOTS, dtype=mx.int32))
        self.assertEqual(mx.abs(foreign_k).max().item(), 0.0)
        self.assertEqual(mx.abs(foreign_v).max().item(), 0.0)

    def test_release_syncs_owned_prefix_and_nothing_after(self):
        runner = self._decoded_runner()
        worker = MlxTpModelWorker.__new__(MlxTpModelWorker)
        worker._mlx_runner = runner
        req = SimpleNamespace(
            rid="r",
            kv=SimpleNamespace(mamba_last_track_seqlen=None),
            owned_kv_len=lambda: len(OWN_SLOTS),
        )
        worker.prepare_for_kv_cache_release(req)
        # The row goes back to the scheduler; the request's MLX state is
        # dropped later, when it leaves the running batch.
        runner.remove_request("r")

        own_k, _ = runner._attention_kv_pool.get_kv(
            0, mx.array([OWN_SLOTS[-1]], dtype=mx.int32)
        )
        self.assertGreater(mx.abs(own_k).max().item(), 0.0)
        self._assert_foreign_slots_untouched(runner)

    def test_retracted_request_writes_nothing(self):
        # A retraction frees the row with no release hook; dropping the
        # state afterwards must not write through the (reusable) row.
        runner = self._decoded_runner()
        runner.remove_request("r")
        self._assert_foreign_slots_untouched(runner)


if __name__ == "__main__":
    unittest.main()
