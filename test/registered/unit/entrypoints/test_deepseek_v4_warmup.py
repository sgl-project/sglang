"""--warmups deepseek_v4_serving_shapes: the startup requests reach every shape class the
serving path compiles a kernel variant for, then flush the cache."""

import asyncio
import unittest
from types import SimpleNamespace
from unittest import mock

from sglang.srt.entrypoints import deepseek_v4_warmup as dsv4_warmup
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class _FakeTokenizerManager:
    def __init__(self, context_len=1 << 20, flush_failures=0):
        self.context_len = context_len
        self.requests = []
        self.groups = []
        self.flushes = 0
        self._flush_failures = flush_failures

    async def generate_request(self, obj, request):
        if isinstance(obj.rid, list):
            total = sum(len(ids) for ids in obj.input_ids)
            self.groups.append((len(obj.rid), total % 16 == 0))
            for rid, ids in zip(obj.rid, obj.input_ids):
                self.requests.append(
                    SimpleNamespace(
                        rid=rid,
                        input_ids=ids,
                        sampling_params=obj.sampling_params,
                        stream=False,
                    )
                )
        else:
            self.requests.append(obj)
        for _ in range(3 if obj.stream else 1):
            await asyncio.sleep(0)
            yield {}

    async def flush_cache(self):
        self.flushes += 1
        return SimpleNamespace(success=self.flushes > self._flush_failures)


def _run(tm, *, mixed_chunk=True, prefill_graph_context=None):
    asyncio.run(
        dsv4_warmup.run_serving_shapes(
            tm, mixed_chunk=mixed_chunk, prefill_graph_context=prefill_graph_context
        )
    )
    return [len(r.input_ids) for r in tm.requests]


class TestDeepseekV4ServingShapes(CustomTestCase):
    def test_prefill_shape_classes(self):
        tm = _FakeTokenizerManager()
        lengths = _run(tm)
        rids = [r.rid for r in tm.requests]
        self.assertEqual(len(set(rids)), len(rids))
        # Triton classes (1, divisible by 16, other) below and above the mHC 4,096-row switch.
        self.assertIn(1, lengths)
        self.assertTrue(any(n % 16 == 0 and 16 <= n < 4096 for n in lengths))
        self.assertTrue(any(1 < n < 16 for n in lengths))
        self.assertIn(4096, lengths)
        self.assertTrue(any(n > 4096 and n % 16 for n in lengths))
        self.assertGreater(max(lengths), 16384)  # chunks with a prefix
        # Prefix hits with a one-token, a short and a >= 4K extend.
        extends = {
            len(s.input_ids) - len(r.input_ids)
            for r in tm.requests
            for s in tm.requests
            if s is not r
            and len(s.input_ids) >= len(r.input_ids)
            and s.input_ids[0] == r.input_ids[0]
            and s.input_ids[: len(r.input_ids)] == r.input_ids
        }
        self.assertIn(0, extends)
        self.assertTrue(any(0 < e < 16 for e in extends))
        self.assertTrue(any(e >= 4096 for e in extends))
        temps = {r.sampling_params.get("temperature") for r in tm.requests}
        self.assertEqual(temps, {None, 0.0})
        self.assertEqual(tm.flushes, 1)

    def test_every_batch_size_in_both_token_count_classes(self):
        """Kernels keyed on next_power_of_2(bs) also specialize the total token count;
        a batch of 9-15 requests with a multiple-of-16 total compiled while serving."""
        tm = _FakeTokenizerManager()
        _run(tm)
        self.assertEqual(
            sorted(tm.groups), [(n, d) for n in range(1, 17) for d in (False, True)]
        )

    def test_mixed_batches_only_with_mixed_chunk(self):
        tm = _FakeTokenizerManager()
        _run(tm)
        streams = [r for r in tm.requests if r.stream]
        self.assertEqual(len(streams), 3)
        tm = _FakeTokenizerManager()
        _run(tm, mixed_chunk=False)
        self.assertFalse(any(r.stream for r in tm.requests))

    def test_prompt_past_the_prefill_graph_context(self):
        """Past the graph's context bucket prefill runs eager, and its row-count
        variants compiled inside the first request longer than the bucket."""
        tm = _FakeTokenizerManager()
        lengths = _run(tm, prefill_graph_context=262144)
        base = 262144 + 8192 + 1001
        self.assertEqual(lengths.count(base), 1)
        self.assertEqual(
            sorted(n - base for n in lengths if n > base), [211, 2064, 5000]
        )
        long = next(r.input_ids for r in tm.requests if len(r.input_ids) == base)
        for r in tm.requests:
            if len(r.input_ids) > base:
                self.assertEqual(r.input_ids[:base], long)
        self.assertLess(max(_run(_FakeTokenizerManager())), 262144)
        short_model = _FakeTokenizerManager(context_len=270000)
        self.assertLess(max(_run(short_model, prefill_graph_context=262144)), 262144)

    def test_graph_context_is_the_tighter_bound(self):
        from sglang.srt.model_executor.cuda_graph_config import Backend

        def context(backend, max_context_size, max_seq_len):
            prefill = SimpleNamespace(
                backend=backend,
                max_context_size=max_context_size,
                max_seq_len=max_seq_len,
            )
            exec_bag = SimpleNamespace(
                graph=SimpleNamespace(
                    cuda_graph_config=SimpleNamespace(prefill=prefill)
                )
            )
            with mock.patch(
                "sglang.srt.runtime_context.get_exec", return_value=exec_bag
            ):
                return dsv4_warmup._prefill_graph_context()

        self.assertEqual(context(Backend.BREAKABLE, 262144, 16384), 16384)
        self.assertEqual(context(Backend.BREAKABLE, None, 262144), 262144)
        self.assertIsNone(context(Backend.BREAKABLE, None, None))
        self.assertIsNone(context(Backend.DISABLED, 262144, 262144))

    def test_flush_is_retried_then_required(self):
        with mock.patch.object(dsv4_warmup.asyncio, "sleep", _no_wait()):
            tm = _FakeTokenizerManager(flush_failures=2)
            _run(tm)
            self.assertEqual(tm.flushes, 3)
            with self.assertRaisesRegex(RuntimeError, "flush"):
                _run(_FakeTokenizerManager(flush_failures=10**6))

    def test_skipped_under_disaggregation(self):
        tm = _FakeTokenizerManager()
        asyncio.run(dsv4_warmup.warm_serving_shapes("prefill", tm))
        self.assertEqual((tm.requests, tm.flushes), ([], 0))


def _no_wait():
    real_sleep = asyncio.sleep

    async def sleep(delay, *args, **kwargs):
        return await real_sleep(0)

    return sleep


if __name__ == "__main__":
    unittest.main()
