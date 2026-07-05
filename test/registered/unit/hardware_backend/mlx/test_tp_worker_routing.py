"""Unit tests for MLX extend-batch request routing in ``MlxTpModelWorker``.

Regression guard: a chunked-prefill *continuation* whose final chunk is
exactly one token (prompt length == k * chunked_prefill_size + 1) must be
routed to the **extend** path, not the **decode** path.

The old routing keyed on ``seq_len > 1`` as a proxy for "is this a
continuation": a 1-token continuation looks identical, by length, to a
genuine single-token decode step mixed into the batch, so it was misrouted
to decode.  The decode path ignores the batch's real token and instead feeds
the model its own stored prediction from the previous chunk -> the true last
prompt token is silently dropped and generation is conditioned on a corrupted
prompt.  The correct discriminator is ``batch.decoding_reqs`` (the genuine
decode reqs mixed in by the scheduler), not the chunk length.

These tests mock the MLX runner and drive the routing logic directly, so they
load no model and run in milliseconds.  They are Apple-Silicon-only only
because ``tp_worker`` imports ``mlx.core`` at module load.
"""

from __future__ import annotations

import importlib.util
import platform
import unittest

import torch

from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.test.ci.ci_register import register_cpu_ci

# AST-parsed "this test exists" marker; actual execution is gated by the
# @skipUnless guard below (mirrors test_quantization.py in this directory).
register_cpu_ci(est_time=5, suite="base-a-test-cpu")

_IS_APPLE_SILICON = platform.system() == "Darwin" and platform.machine() == "arm64"
_HAS_MLX = importlib.util.find_spec("mlx") is not None
_SKIP_REASON = "Apple-Silicon-only (tp_worker imports mlx.core at module load)"


class _FakeRunner:
    """Records which routing path each request took."""

    def __init__(self, known_rids):
        self._known = set(known_rids)
        self.calls: list[tuple[str, str]] = []  # (op, rid)
        self._counter = 0

    def flush_all_decode_kv(self):
        pass

    def has_request(self, rid):
        return rid in self._known

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
    ):
        self.calls.append(("prefill", req_id))
        return 3000

    def ops_for(self, rid):
        return [op for op, r in self.calls if r == rid]


class _FakeReq:
    def __init__(self, rid, req_pool_idx=0):
        self.rid = rid
        self.prefix_indices = torch.empty(0, dtype=torch.long)
        self.fill_ids = [0]
        self.req_pool_idx = req_pool_idx


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
    """Routing contract for ``MlxTpModelWorker._forward_batch_generation_mlx``."""

    @staticmethod
    def _run(reqs, extend_lens, known_rids, decoding_reqs, forward_mode):
        from sglang.srt.hardware_backend.mlx.tp_worker import MlxTpModelWorker

        runner = _FakeRunner(known_rids)
        worker = MlxTpModelWorker.__new__(MlxTpModelWorker)
        worker._mlx_runner = runner
        worker._mlx_active_rids = set()

        batch = _FakeBatch(forward_mode, reqs, extend_lens, decoding_reqs)
        result = worker._forward_batch_generation_mlx(batch)
        # Every request must be assigned exactly one output token.
        assert result.next_token_ids.numel() == len(reqs)
        return runner

    def test_one_token_continuation_routes_to_extend(self):
        """THE REGRESSION: a 1-token continuation must extend, not decode."""
        req = _FakeReq("r1")
        runner = self._run(
            reqs=[req],
            extend_lens=[1],  # final continuation chunk = 1 token
            known_rids={"r1"},  # already prefilled an earlier chunk
            decoding_reqs=None,  # not a genuine decode
            forward_mode=ForwardMode.EXTEND,
        )
        self.assertEqual(runner.ops_for("r1"), ["extend"])
        self.assertNotIn("decode", runner.ops_for("r1"))

    def test_multi_token_continuation_routes_to_extend(self):
        """Unchanged behavior: a multi-token continuation extends."""
        req = _FakeReq("r1")
        runner = self._run(
            reqs=[req],
            extend_lens=[4],
            known_rids={"r1"},
            decoding_reqs=None,
            forward_mode=ForwardMode.EXTEND,
        )
        self.assertEqual(runner.ops_for("r1"), ["extend"])

    def test_genuine_mixed_decode_still_routes_to_decode(self):
        """Safety: a real single-token decode mixed into the batch must decode.

        This is the case the old ``seq_len > 1`` check got *right* and that the
        fix must not break: the discriminator is membership in decoding_reqs.
        """
        prefill_req = _FakeReq("p1")  # new prefill chunk
        decode_req = _FakeReq("d1")  # genuine decode step, already prefilled
        runner = self._run(
            reqs=[prefill_req, decode_req],
            extend_lens=[4, 1],  # d1's decode step is 1 token
            known_rids={"d1"},  # d1 already fully prefilled
            decoding_reqs=[decode_req],  # scheduler marks d1 as a decode
            forward_mode=ForwardMode.MIXED,
        )
        self.assertEqual(runner.ops_for("p1"), ["prefill"])
        self.assertEqual(runner.ops_for("d1"), ["decode"])
        self.assertNotIn("extend", runner.ops_for("d1"))

    def test_one_token_decode_vs_continuation_only_differ_by_decoding_reqs(self):
        """Same length (1 token), opposite routing, decided solely by decoding_reqs."""
        # In decoding_reqs -> decode.
        d = _FakeReq("x")
        r_decode = self._run([d], [1], {"x"}, [d], ForwardMode.MIXED)
        self.assertEqual(r_decode.ops_for("x"), ["decode"])
        # Not in decoding_reqs -> extend (the continuation).
        c = _FakeReq("x")
        r_ext = self._run([c], [1], {"x"}, None, ForwardMode.EXTEND)
        self.assertEqual(r_ext.ops_for("x"), ["extend"])


if __name__ == "__main__":
    unittest.main()
