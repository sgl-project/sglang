"""Unit tests for MLX extend-batch request routing in ``MlxTpModelWorker``.

Regression guard for the single-token chunked-prefill continuation bug: a
continuation whose final chunk is exactly one token (prompt length ==
k * chunked_prefill_size + 1) must be routed to the **extend** path, not the
**decode** path.

The old routing keyed on ``seq_len > 1`` as a proxy for "is this a
continuation"; a 1-token continuation is indistinguishable, by length, from a
genuine single-token decode step mixed into the batch, so it was misrouted to
decode. The decode path ignores the batch's real token and feeds the model its
own stored prediction from the previous chunk -> the true last prompt token is
silently dropped and generation is conditioned on a corrupted prompt. The
correct discriminator is ``batch.decoding_reqs``, not the chunk length.

The routing decision was duplicated across the sync and async paths (the bug
therefore existed in both). It now lives in the shared
``MlxTpModelWorker._route_extend_request`` helper. These tests cover:

  * the helper decision directly (both paths delegate to it);
  * the sync wiring, by driving ``_forward_batch_generation_mlx``;
  * the async wiring, by driving ``_async_extend_batch``.

They mock the MLX runner and load no model. Apple-Silicon-only because
``tp_worker`` imports ``mlx.core`` at module load.
"""

from __future__ import annotations

import importlib.util
import platform
import unittest
from types import SimpleNamespace

import torch

from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.test.ci.ci_register import register_cpu_ci, register_mlx_ci

# CPU marker is AST-parsed "this test exists"; actual CPU-side execution is
# gated by the @skipUnless guard below. MLX marker runs for real on the MLX
# lane's stage-a (model-free: mocks the runner, loads no model).
register_cpu_ci(est_time=10, suite="base-a-test-cpu")
register_mlx_ci(est_time=10, suite="stage-a-unit-test-mlx")

_IS_APPLE_SILICON = platform.system() == "Darwin" and platform.machine() == "arm64"
_HAS_MLX = importlib.util.find_spec("mlx") is not None
_SKIP_REASON = "Apple-Silicon-only (tp_worker imports mlx.core at module load)"


class _FakeRunner:
    """Records which routing path each request took (sync + async surfaces)."""

    def __init__(self, known_rids):
        self._known = set(known_rids)
        self.calls: list[tuple[str, str]] = []  # (op, rid)
        self._req_caches: dict[str, list] = {}
        self._counter = 0

    # --- shared ---
    def has_request(self, rid):
        return rid in self._known

    def flush_all_decode_kv(self):
        pass

    def ops_for(self, rid):
        return [op for op, r in self.calls if r == rid]

    @staticmethod
    def _fake_cache_layer():
        import mlx.core as mx

        return SimpleNamespace(state=[mx.array([0.0], dtype=mx.float32)])

    # --- sync surface ---
    def extend(self, rid, new_token_ids, new_slot_ids):
        self.calls.append(("extend", rid))
        self._counter += 1
        return 1000 + self._counter

    def decode_batch(self, rids):
        for rid in rids:
            self.calls.append(("decode", rid))
        return [2000 + i for i in range(len(rids))]

    def prefill(
        self,
        req_id,
        new_token_ids,
        full_token_ids,
        prefix_slot_ids,
        new_slot_ids,
        req_pool_idx,
        req=None,
    ):
        self.calls.append(("prefill", req_id))
        return 3000

    # --- async surface ---
    def extend_start(self, req_id, new_token_ids, new_slot_ids):
        import mlx.core as mx

        self.calls.append(("extend_start", req_id))
        self._req_caches[req_id] = [self._fake_cache_layer()]
        return SimpleNamespace(lazy_token=mx.array([0], dtype=mx.int32), req_id=req_id)

    def prefill_start(
        self,
        req_id,
        new_token_ids,
        full_token_ids,
        prefix_slot_ids,
        new_slot_ids,
        req_pool_idx,
        req=None,
    ):
        import mlx.core as mx

        self.calls.append(("prefill_start", req_id))
        return SimpleNamespace(
            lazy_token=mx.array([0], dtype=mx.int32),
            cache=[self._fake_cache_layer()],
            req_id=req_id,
        )

    def decode_batch_start(self, rids):
        import mlx.core as mx

        for rid in rids:
            self.calls.append(("decode_start", rid))
        return SimpleNamespace(
            lazy_tokens=mx.array([0] * len(rids), dtype=mx.int32),
            caches=[[self._fake_cache_layer()] for _ in rids],
            req_ids=list(rids),
        )


class _FakeReq:
    def __init__(self, rid, req_pool_idx=0):
        self.rid = rid
        self.prefix_indices = torch.empty(0, dtype=torch.long)
        self.fill_ids = [0]
        self.req_pool_idx = req_pool_idx

    def get_fill_ids(self):
        return self.fill_ids


class _FakeBatch:
    def __init__(self, forward_mode, reqs, extend_lens, decoding_reqs=None):
        total = sum(extend_lens)
        self.forward_mode = forward_mode
        self.reqs = reqs
        self.extend_lens = list(extend_lens)
        self.decoding_reqs = decoding_reqs
        # Arbitrary but correctly-sized token / slot arrays.
        self.input_ids = torch.arange(total, dtype=torch.long)
        self.out_cache_loc = torch.arange(total, dtype=torch.long)


@unittest.skipUnless(_IS_APPLE_SILICON and _HAS_MLX, _SKIP_REASON)
class TestMlxExtendRouting(unittest.TestCase):
    """Routing contract for MlxTpModelWorker: shared helper + sync + async."""

    @staticmethod
    def _worker(known_rids):
        from sglang.srt.hardware_backend.mlx.tp_worker import MlxTpModelWorker

        worker = MlxTpModelWorker.__new__(MlxTpModelWorker)
        worker._mlx_runner = _FakeRunner(known_rids)
        worker._mlx_active_rids = set()
        return worker

    # ---------- the shared decision helper ----------
    # The helper takes no seq_len: length cannot distinguish a 1-token
    # continuation from a genuine decode -- request state does.

    def test_route_unseen_request_is_prefill(self):
        worker = self._worker(known_rids=set())
        self.assertEqual(worker._route_extend_request("r1", set()), "prefill")

    def test_route_seen_non_decode_is_continuation(self):
        worker = self._worker(known_rids={"r1"})
        self.assertEqual(worker._route_extend_request("r1", set()), "continuation")

    def test_route_seen_and_in_decoding_reqs_is_decode(self):
        worker = self._worker(known_rids={"r1"})
        self.assertEqual(worker._route_extend_request("r1", {"r1"}), "decode")

    # ---------- sync path: _forward_batch_generation_mlx ----------

    def _run_sync(self, reqs, extend_lens, known_rids, decoding_reqs, forward_mode):
        worker = self._worker(known_rids)
        batch = _FakeBatch(forward_mode, reqs, extend_lens, decoding_reqs)
        result = worker._forward_batch_generation_mlx(batch)
        assert result.next_token_ids.numel() == len(reqs)
        return worker._mlx_runner

    def test_sync_one_token_continuation_routes_to_extend(self):
        """THE REGRESSION (sync): a 1-token continuation must extend, not decode."""
        runner = self._run_sync([_FakeReq("r1")], [1], {"r1"}, None, ForwardMode.EXTEND)
        self.assertEqual(runner.ops_for("r1"), ["extend"])

    def test_sync_genuine_mixed_decode_routes_to_decode(self):
        p, d = _FakeReq("p1"), _FakeReq("d1")
        runner = self._run_sync([p, d], [4, 1], {"d1"}, [d], ForwardMode.MIXED)
        self.assertEqual(runner.ops_for("p1"), ["prefill"])
        self.assertEqual(runner.ops_for("d1"), ["decode"])

    # ---------- async path: _async_extend_batch ----------

    def _run_async(self, reqs, extend_lens, known_rids, decoding_reqs, forward_mode):
        from sglang.srt.hardware_backend.mlx.tp_worker import MlxTpModelWorker

        worker = MlxTpModelWorker.__new__(MlxTpModelWorker)
        worker._mlx_runner = _FakeRunner(known_rids)
        batch = _FakeBatch(forward_mode, reqs, extend_lens, decoding_reqs)
        # returns (lazy_stacked, pending_prefills, pending_extends,
        #          pending_mixed_decode, mode)
        result = worker._async_extend_batch(batch)
        return worker._mlx_runner, result

    def test_async_one_token_continuation_routes_to_extend(self):
        """THE REGRESSION (async): a 1-token continuation must extend, not decode."""
        runner, result = self._run_async(
            [_FakeReq("r1")], [1], {"r1"}, None, ForwardMode.EXTEND
        )
        self.assertEqual(runner.ops_for("r1"), ["extend_start"])
        self.assertEqual(len(result[2]), 1)  # one pending extend
        self.assertIsNone(result[3])  # no mixed decode

    def test_async_genuine_mixed_decode_routes_to_decode(self):
        p, d = _FakeReq("p1"), _FakeReq("d1")
        runner, result = self._run_async([p, d], [4, 1], {"d1"}, [d], ForwardMode.MIXED)
        self.assertEqual(runner.ops_for("p1"), ["prefill_start"])
        self.assertEqual(runner.ops_for("d1"), ["decode_start"])
        self.assertIsNotNone(result[3])  # pending mixed decode present


if __name__ == "__main__":
    unittest.main()
