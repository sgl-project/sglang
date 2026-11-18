"""Synthetic chunk profiling must stay within request bounds and release on failure."""

import unittest
import weakref
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.managers.scheduler_components import dynamic_chunk_sizer as module
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=11, suite="base-a-test-cpu")


class TestDynamicChunkSizerProfiling(CustomTestCase):
    def _profile(self, *, context, mapping, prefill, base, fail_forward=False):
        prepared = []
        live = set()
        runner = SimpleNamespace(
            model_config=SimpleNamespace(
                context_len=context, hidden_size=2, dtype=torch.float32
            ),
            get_pp_proxy_topk_size=lambda: None,
            get_pp_proxy_residual_num_blocks=lambda: None,
        )

        def make_req(**kwargs):
            return SimpleNamespace(
                rid=kwargs["rid"],
                full_untruncated_fill_ids=kwargs["origin_input_ids"],
                init_next_round_input=Mock(),
                last_node=None,
                kv=SimpleNamespace(holds_kv=False),
            )

        def make_batch(reqs, *_args):
            req = reqs[0]

            def prepare():
                prepared.append(req.extend_end)
                live.add(req.rid)
                req.kv.holds_kv = True

            return SimpleNamespace(
                prepare_for_extend=prepare,
                input_ids=torch.zeros(1, dtype=torch.int64),
                forward_mode=SimpleNamespace(is_extend=lambda: True),
            )

        def release(req, _cache, *, checkpoint):
            self.assertFalse(checkpoint)
            live.remove(req.rid)
            req.kv.holds_kv = False

        runner.forward = Mock(
            side_effect=RuntimeError("synthetic forward failed")
            if fail_forward
            else None
        )
        with (
            patch.object(
                module, "get_parallel", return_value=SimpleNamespace(pp_rank=0)
            ),
            patch.object(module, "Req", side_effect=make_req),
            patch.object(module.ScheduleBatch, "init_new", side_effect=make_batch),
            patch.object(
                module.ForwardBatch, "init_new", return_value=SimpleNamespace()
            ),
            patch.object(module, "release_kv_cache", side_effect=release),
            patch.object(module, "is_dp_attention_enabled", return_value=False),
            patch.object(module, "set_is_extend_in_batch"),
            patch.object(module, "get_device_module", return_value=Mock()),
            patch.object(module, "tqdm", side_effect=lambda items, **_kwargs: items),
        ):
            sizer = module.DynamicChunkSizer(
                model_runner=runner,
                model_config=runner.model_config,
                tree_cache=Mock(),
                req_to_token_pool=SimpleNamespace(max_context_len=mapping),
                token_to_kv_pool_allocator=Mock(),
                spec_algorithm=Mock(),
                chunked_prefill_size=base,
                max_prefill_tokens=prefill,
                page_size=1,
                device="cpu",
            )
            if fail_forward:
                with self.assertRaisesRegex(RuntimeError, "synthetic forward failed"):
                    sizer._profile_prefill_latency()
                self.assertEqual(len(prepared), 1)
            else:
                lengths, latencies = sizer._profile_prefill_latency()
                self.assertEqual(lengths, prepared)
                self.assertEqual(len(latencies), len(lengths))
                self.assertGreater(len(set(lengths)), 8)
                self.assertGreater(min(lengths), 0)
                self.assertLessEqual(max(lengths), min(context, mapping))
            self.assertEqual(
                live, set(), "profiling left a prepared request holding KV"
            )
            return prepared

    def test_samples_fit_request_capacity_and_preserve_probe_range(self):
        for context, mapping, prefill, base, expected_max in (
            (8192, 8196, 16384, 16384, 8192),
            (8192, 4096, 16384, 16384, 4096),
            (8192, 8196, 2048, 16384, 8192),
            (8192, 8196, 16384, 512, 640),
            (32768, 32772, 16384, 16384, 20480),
        ):
            with self.subTest(
                context=context, mapping=mapping, prefill=prefill, base=base
            ):
                lengths = self._profile(
                    context=context, mapping=mapping, prefill=prefill, base=base
                )
                self.assertEqual(max(lengths), expected_max)

    def test_forward_failure_releases_prepared_request(self):
        self._profile(
            context=8192, mapping=8196, prefill=16384, base=512, fail_forward=True
        )


class TestDynamicChunkSizerCacheRelease(CustomTestCase):
    """PP0 hands the probes' cached memory back before serving, whether or not they fit.

    Left cached, it starves allocations made outside the framework allocator (oneCCL,
    Level Zero) and the first request hangs or fails; after a failed probe the release
    frees nothing while the handler's traceback still holds the probe's activations.
    """

    def _profile_and_fit(self, *, probe_fails):
        probe_refs = []
        probe_freed_at_release = []

        def probe():
            activations = torch.empty(8)
            probe_refs.append(weakref.ref(activations))
            if probe_fails:
                raise torch.OutOfMemoryError("probe ran out of memory")
            lengths = list(range(4096, 1024, -256))
            return lengths, [1e-6 * n * n + 1e-2 * n + 5 for n in lengths]

        def empty_cache():
            probe_freed_at_release.append(probe_refs[0]() is None)

        parallel = SimpleNamespace(
            pp_rank=0,
            pp_group=SimpleNamespace(is_first_rank=True),
            world_group=SimpleNamespace(rank=0, cpu_group=None),
        )
        with (
            patch.object(module, "get_parallel", return_value=parallel),
            patch.object(
                module, "broadcast_pyobj", side_effect=lambda data, *_a, **_k: data
            ),
            patch.object(
                module.current_platform, "empty_cache", side_effect=empty_cache
            ),
        ):
            sizer = module.DynamicChunkSizer(
                model_runner=Mock(),
                model_config=Mock(),
                tree_cache=Mock(),
                req_to_token_pool=Mock(),
                token_to_kv_pool_allocator=Mock(),
                spec_algorithm=Mock(),
                chunked_prefill_size=2048,
                max_prefill_tokens=16384,
                page_size=1,
                device="cpu",
            )
            with patch.object(sizer, "_profile_prefill_latency", side_effect=probe):
                ready = sizer.profile_and_fit()
        return ready, probe_freed_at_release

    def test_probe_memory_is_released_after_profiling(self):
        for probe_fails in (False, True):
            with self.subTest(probe_fails=probe_fails):
                ready, freed = self._profile_and_fit(probe_fails=probe_fails)
                self.assertEqual(ready, not probe_fails)
                self.assertEqual(freed, [True])


class TestChunkSizePredictorSmoothing(CustomTestCase):
    """The smoothing factor and floor semantics the PP long-context guide promises."""

    BASE = 12288
    PAGE = 128
    # Coefficients fitted on Llama-3.1-8B PP4 on Intel Arc Pro B60 (ms, tokens).
    COEFFS = (1.18e-06, 3.85e-02, 22.5)
    HISTORY = (12288, 32768, 65536, 98304, 131072, 262144, 524288)

    def _predictor(self):
        a, b, c = self.COEFFS
        # Same probe shape as the profiler: 128 descending lengths up to 1.25x base.
        top = int(self.BASE * 1.25)
        lengths = [top - i * (top // 128) for i in range(128)]
        predictor = module.ChunkSizePredictor()
        predictor.fit(lengths, [a * n * n + b * n + c for n in lengths])
        predictor.set_target_latency(self.BASE)
        predictor.is_ready = True
        return predictor

    def _schedule(self, smooth_factor):
        predictor = self._predictor()
        with module.envs.SGLANG_DYNAMIC_CHUNKING_SMOOTH_FACTOR.override(smooth_factor):
            return [
                predictor.predict_next_chunk_size(
                    history_len=history,
                    base_chunk_size=self.BASE,
                    page_size=self.PAGE,
                    context_len=1 << 20,
                    max_chunk_size=16384,
                )
                for history in self.HISTORY
            ]

    def test_zero_smoothing_keeps_the_fixed_chunk_size(self):
        self.assertEqual(self._schedule(0.0), [self.BASE] * len(self.HISTORY))

    def test_chunks_shrink_with_history_down_to_a_quarter_of_the_base(self):
        for smooth_factor in (0.65, 0.75, 1.0):
            with self.subTest(smooth_factor=smooth_factor):
                sizes = self._schedule(smooth_factor)
                self.assertEqual(sizes, sorted(sizes, reverse=True))
                self.assertLess(sizes[-1], self.BASE)
                for size in sizes:
                    self.assertEqual(size % self.PAGE, 0)
                    self.assertGreaterEqual(size, self.BASE // 4)
        # Strict smoothing reaches the floor well inside a 512K prefix.
        self.assertEqual(self._schedule(1.0)[-1], self.BASE // 4)

    def test_larger_smoothing_factor_shrinks_chunks_faster(self):
        gentle, default, strict = (self._schedule(s) for s in (0.65, 0.75, 1.0))
        for g, d, s in zip(gentle, default, strict):
            self.assertGreaterEqual(g, d)
            self.assertGreaterEqual(d, s)
        self.assertGreater(sum(gentle), sum(default))
        self.assertGreater(sum(default), sum(strict))


if __name__ == "__main__":
    unittest.main()
