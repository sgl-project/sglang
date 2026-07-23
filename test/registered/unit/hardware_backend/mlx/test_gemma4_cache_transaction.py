from __future__ import annotations

import importlib.util
import unittest

from sglang.test.ci.ci_register import register_mlx_ci

register_mlx_ci(est_time=2, suite="stage-a-unit-test-mlx")

_HAS_MLX = (
    importlib.util.find_spec("mlx") is not None
    and importlib.util.find_spec("mlx_lm.models.gemma4_text") is not None
)

if _HAS_MLX:
    import mlx.core as mx
    from registered.unit.hardware_backend.mlx.gemma4_test_utils import (
        assert_native_cache_equal,
        cache_logical_length,
        native_cache_snapshot,
        tiny_gemma4,
    )

    from sglang.srt.hardware_backend.mlx.kv_cache.native_transaction import (
        MlxNativeCacheTransaction,
        clone_native_cache,
    )
    from sglang.srt.hardware_backend.mlx.spec_decode import (
        MlxVerifySegment,
        verify_greedy_segment,
    )


@unittest.skipUnless(_HAS_MLX, "requires MLX and mlx-lm Gemma 4")
class TestGemma4NativeCacheTransaction(unittest.TestCase):
    def setUp(self):
        self.model = tiny_gemma4()

    def _forward(self, cache, tokens):
        return self.model(mx.array([tokens], dtype=mx.int32), cache=cache)

    def _prefill(self, prompt):
        cache = self.model.make_cache()
        logits = self._forward(cache, tuple(prompt))
        root = mx.argmax(logits[:, -1, :], axis=-1)
        mx.eval(root, *[item for entry in cache for item in entry.state])
        return cache, int(root.item())

    def _correct_draft(self, cache, root):
        probe = clone_native_cache(cache)
        logits = self._forward(probe, (root,))
        token = mx.argmax(logits[:, -1, :], axis=-1)
        mx.eval(token)
        return int(token.item())

    def _exercise(self, prompt, *, accept):
        cache, root = self._prefill(prompt)
        draft = self._correct_draft(cache, root)
        if not accept:
            draft = (draft + 1) % self.model.args.vocab_size

        transaction = MlxNativeCacheTransaction(cache, (root, draft), self._forward)
        speculative = transaction.begin()
        logits = self._forward(speculative, (root, draft))
        target_ids = tuple(
            int(item) for item in mx.argmax(logits, axis=-1).reshape(-1).tolist()
        )
        decision = verify_greedy_segment(
            MlxVerifySegment("r", (draft,), 1, target_ids, 0, 2)
        )
        transaction.commit(decision.committed_query_count)

        reference, reference_root = self._prefill(prompt)
        self.assertEqual(root, reference_root)
        self._forward(reference, (root, draft)[: decision.committed_query_count])
        mx.eval(*[item for entry in reference for item in entry.state])
        assert_native_cache_equal(self, cache, reference)

        token_history = list(prompt) + [root] + list(decision.emitted_token_ids)
        self.assertEqual(len(token_history), cache_logical_length(cache) + 1)
        return cache

    def test_accept_and_reject_match_target_only(self):
        self._exercise([1, 2, 3], accept=True)
        self._exercise([1, 2, 3], accept=False)

    def test_before_at_after_and_multiple_window_rotations(self):
        for prompt_len in (7, 8, 9, 17, 25):
            for accept in (False, True):
                with self.subTest(prompt_len=prompt_len, accept=accept):
                    prompt = [1 + (index % 29) for index in range(prompt_len)]
                    cache = self._exercise(prompt, accept=accept)
                    self.assertEqual(len(cache), 2)  # compact YOCO owner list
                    self.assertEqual(
                        [type(entry).__name__ for entry in cache],
                        ["RotatingKVCache", "KVCache"],
                    )

    def test_abort_and_forward_failure_restore_original(self):
        cache, root = self._prefill(list(range(1, 12)))
        before = clone_native_cache(cache)
        draft = self._correct_draft(cache, root)

        transaction = MlxNativeCacheTransaction(cache, (root, draft), self._forward)
        speculative = transaction.begin()
        self._forward(speculative, (root, draft))
        transaction.abort()
        assert_native_cache_equal(self, cache, before)

        def failing_replay(target_cache, tokens):
            result = self._forward(target_cache, tokens)
            mx.eval(result, *[item for entry in target_cache for item in entry.state])
            raise RuntimeError("synthetic target failure")

        transaction = MlxNativeCacheTransaction(cache, (root,), failing_replay)
        transaction.begin()
        with self.assertRaisesRegex(RuntimeError, "synthetic target failure"):
            transaction.commit(1)
        assert_native_cache_equal(self, cache, before)

    def test_transactions_are_single_use_and_request_isolated(self):
        cache_a, root_a = self._prefill([1, 2])
        cache_b, _ = self._prefill([7, 8, 9])
        before_b = native_cache_snapshot(cache_b)
        transaction = MlxNativeCacheTransaction(cache_a, (root_a,), self._forward)
        transaction.begin()
        with self.assertRaisesRegex(RuntimeError, "single-use"):
            transaction.begin()
        transaction.commit(1)
        with self.assertRaisesRegex(RuntimeError, "active"):
            transaction.commit(1)
        self.assertEqual(transaction.committed_count, 1)

        after_b = native_cache_snapshot(cache_b)
        for left, right in zip(before_b, after_b):
            self.assertEqual(left["offset"], right["offset"])
            self.assertTrue((left["keys"] == right["keys"]).all())

        abort_txn = MlxNativeCacheTransaction(cache_b, (0,), self._forward)
        abort_txn.begin()
        abort_txn.abort()
        with self.assertRaisesRegex(RuntimeError, "active"):
            abort_txn.abort()


if __name__ == "__main__":
    unittest.main()
