"""Regression guard for the MLX chained-decode pool-sync bug (#42415).

Chained decode steps skip the scheduler's per-step allocation entirely, so
a request's ``req_to_token`` row only ever gains ONE decode slot (from the
first, freshly-launched decode step). The tree insert at finish is capped
at ``owned_kv_len`` accordingly. The decode-KV flush, though, keyed its
range off the MLX cache offset, which counts every decoded token. Reading
``req_to_token[row, synced:cache_offset]`` therefore ran past the owned
prefix into unwritten (zero) or recycled (another request's) row content,
and scattered this request's KV into slots owned by the tree or by the
next request reusing the row. A repeat of an earlier chat prompt then
matched poisoned tree slots and answered from the wrong conversation.

The flush is now driven from ``prepare_for_kv_cache_release`` (row still
owned, release follows immediately) and clamped to ``owned_kv_len``; the
request is then sealed so the deferred flush in ``remove_request`` no-ops.

These tests mock the runner / pools and load no model. Apple-Silicon-only
because ``tp_worker`` imports ``mlx.core`` at module load.
"""

from __future__ import annotations

import importlib.util
import platform
import unittest
from types import SimpleNamespace

from sglang.srt.managers.schedule_batch import ReqKvInfo
from sglang.test.ci.ci_register import register_mlx_ci
from sglang.test.test_utils import CustomTestCase

register_mlx_ci(est_time=2, suite="stage-a-unit-test-mlx")

_IS_APPLE_SILICON = platform.system() == "Darwin" and platform.machine() == "arm64"
_HAS_MLX = importlib.util.find_spec("mlx") is not None
_SKIP_REASON = "Apple-Silicon-only (tp_worker imports mlx.core at module load)"


class _FakeRunner:
    """Records the release-prep calls the worker makes, in order."""

    def __init__(self, known_rids):
        self._known = set(known_rids)
        self.calls: list[tuple] = []

    def has_request(self, rid):
        return rid in self._known

    def flush_decode_kv_for_request(self, rid, owned_len=None):
        self.calls.append(("flush", rid, owned_len))

    def store_auxiliary_state_for_request(self, rid):
        self.calls.append(("store_aux", rid))


@unittest.skipUnless(_IS_APPLE_SILICON and _HAS_MLX, _SKIP_REASON)
class TestMlxReleaseFlush(CustomTestCase):
    """``prepare_for_kv_cache_release`` flushes the owned prefix, clamped."""

    @staticmethod
    def _worker(known_rids):
        from sglang.srt.hardware_backend.mlx.tp_worker import MlxTpModelWorker

        worker = MlxTpModelWorker.__new__(MlxTpModelWorker)
        worker._mlx_runner = _FakeRunner(known_rids)
        return worker

    @staticmethod
    def _req(rid, owned_len=16):
        return SimpleNamespace(
            rid=rid,
            kv=ReqKvInfo(req_pool_idx=0),
            owned_kv_len=lambda: owned_len,
        )

    def test_prepare_flushes_owned_prefix_before_release(self):
        worker = self._worker(["r1"])
        req = self._req("r1", owned_len=16)

        worker.prepare_for_kv_cache_release(req)

        self.assertEqual(
            worker._mlx_runner.calls,
            [("flush", "r1", 16), ("store_aux", "r1")],
            msg="decode KV must be flushed with the owned_kv_len clamp "
            "ahead of the auxiliary-state snapshot",
        )

    def test_prepare_skips_request_unknown_to_runner(self):
        worker = self._worker(known_rids=[])
        req = self._req("gone")
        req.kv.mamba_last_track_seqlen = 7

        worker.prepare_for_kv_cache_release(req)

        self.assertEqual(worker._mlx_runner.calls, [])
        # Untouched when the runner has no state for the request.
        self.assertEqual(req.kv.mamba_last_track_seqlen, 7)

    def test_flush_clamps_to_owned_prefix_and_seals(self):
        import torch

        from sglang.srt.hardware_backend.mlx.model_runner import MlxModelRunner

        runner = MlxModelRunner.__new__(MlxModelRunner)
        runner.disable_radix_cache = False
        runner._attention_kv_pool = object()
        runner._cache_layout = SimpleNamespace(first_attention_layer_index=0)
        # cache offset counts every decoded token; the row only has the
        # first decode slot written, the rest is stale recycled content.
        runner._req_caches = {"r1": [SimpleNamespace(offset=47)]}
        runner._req_pool_idx = {"r1": 3}
        runner._req_synced_offset = {"r1": 15}
        req_to_token = torch.zeros(4, 64, dtype=torch.long)
        req_to_token[3, 15] = 25
        req_to_token[3, 16:24] = torch.arange(16, 24)
        runner._req_to_token_pool = SimpleNamespace(req_to_token=req_to_token)

        synced = []
        runner._sync_new_kv_to_pool = lambda cache, start, slots: synced.append(
            (start, list(slots))
        )

        runner.flush_decode_kv_for_request("r1", owned_len=16)
        self.assertEqual(
            synced,
            [(15, [25])],
            msg="flush must stop at the owned prefix instead of reading "
            "stale row content as slot ids",
        )
        self.assertEqual(runner._req_synced_offset["r1"], 47)

        # The deferred flush in remove_request is now a no-op.
        runner._sync_decode_kv_to_pool("r1")
        self.assertEqual(len(synced), 1)

    def test_flush_respects_disable_radix_cache(self):
        from sglang.srt.hardware_backend.mlx.model_runner import MlxModelRunner

        runner = MlxModelRunner.__new__(MlxModelRunner)
        runner.disable_radix_cache = True
        runner._attention_kv_pool = None
        runner._req_to_token_pool = None

        # Must return without touching the (absent) pools.
        runner.flush_decode_kv_for_request("r1", owned_len=16)


if __name__ == "__main__":
    unittest.main()
