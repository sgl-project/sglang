# SPDX-License-Identifier: Apache-2.0
"""CPU contracts for concurrency-bounded preparation, not GPU qualification."""

import unittest
from types import SimpleNamespace as NS
from unittest.mock import patch

from sglang.srt.kernel_plugins.nvidia import binding as mb
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, stage="base-a", runner_config="cpu")
ARCH = ["KimiK3ForConditionalGeneration"]
DEFAULT = tuple(range(6, 65, 6))


class TestCapturePreparation(unittest.TestCase):
    def setUp(self):
        self.graph = NS(decode=NS(backend=mb.Backend.FULL, bs=[1, 2]))
        self.config = NS(speculative_algorithm="DSPARK", speculative_adaptive=False)
        self.parallel = NS(
            tp_size=8,
            pp_size=1,
            dp_size=1,
            attn_cp_size=1,
            dcp_size=1,
            enable_dp_attention=False,
        )
        self.schedule = NS(max_running_requests=4)
        self.disagg = NS(disaggregation_mode="null", enable_pd_role_switch=False)
        for name, value in (
            ("get_exec", NS(graph=NS(cuda_graph_config=self.graph))),
            ("get_spec", self.config),
            ("get_parallel", self.parallel),
            ("get_schedule", self.schedule),
            ("get_disagg", self.disagg),
        ):
            context = patch.object(mb, name, return_value=value)
            context.start()
            self.addCleanup(context.stop)
        context = patch(
            "sglang.srt.layers.dp_attention.get_cuda_graph_batch_size_alignment",
            return_value=1,
        )
        self.alignment = context.start()
        self.addCleanup(context.stop)
        context = patch.object(
            mb, "_gather_tp", side_effect=lambda value, **kw: [value] * 8
        )
        self.gather = context.start()
        self.addCleanup(context.stop)

    def select(self, q=6, arch=ARCH, draft=False):
        return mb._kimi_build_widths(64, q, arch, is_draft=draft)

    def test_keeps_eager_widths_and_caps_only_unreachable_rows(self):
        for limit, expected in (
            (1, (6,)),
            (4, (6, 12, 18, 24)),
            (10, DEFAULT),
            (48, DEFAULT),
        ):
            self.schedule.max_running_requests = limit
            for batches in ([1, 2], [16], [1, 4, 8]):
                with self.subTest(limit=limit, batches=batches):
                    self.graph.decode.bs = batches
                    self.assertEqual(self.select(), expected)
                    self.assertEqual(self.graph.decode.bs, batches)

    def test_uncertain_bounds_or_layouts_keep_full_preparation(self):
        for owner, key, values in (
            (self.schedule, "max_running_requests", [None, 0, -1, True, 1.5]),
            (self.config, "speculative_algorithm", [None, "EAGLE"]),
            (self.config, "speculative_adaptive", [True]),
            (self.parallel, "tp_size", [16]),
            (self.parallel, "pp_size", [2]),
            (self.parallel, "dp_size", [2]),
            (self.parallel, "attn_cp_size", [8]),
            (self.parallel, "dcp_size", [8]),
            (self.parallel, "enable_dp_attention", [True]),
            (self.disagg, "disaggregation_mode", ["decode", "prefill"]),
            (self.disagg, "enable_pd_role_switch", [True]),
            (self.graph.decode, "backend", [mb.Backend.DISABLED]),
        ):
            original = getattr(owner, key)
            for value in values:
                with self.subTest(key=key, value=value):
                    setattr(owner, key, value)
                    self.assertEqual(self.select(), DEFAULT)
            setattr(owner, key, original)
        self.alignment.return_value = 8
        self.assertEqual(self.select(), DEFAULT)

    def test_other_models_drafts_and_query_widths_are_unchanged(self):
        self.assertEqual(self.select(arch=["Qwen3_5ForConditionalGeneration"]), DEFAULT)
        self.assertEqual(self.select(draft=True), DEFAULT)
        self.gather.assert_not_called()
        for q in (1, 4, 8):
            self.assertEqual(self.select(q=q), tuple(range(q, 65, q)))

    def test_ranks_must_agree_before_preparation(self):
        self.gather.side_effect = lambda value, **kw: [value] * 7 + [DEFAULT]
        with self.assertRaisesRegex(RuntimeError, "ranks.*differ"):
            self.select()


class TestNativeCapturePreparation(unittest.TestCase):
    def test_request_pool_clamps_oversized_capture_bucket(self):
        from sglang.srt.arg_groups.cuda_graph_hook import parse_cuda_graph_config
        from sglang.srt.arg_groups.memory_hook import handle_gpu_memory_settings
        from sglang.srt.mem_cache.kv_cache_configurator import KVCacheConfigurator
        from sglang.srt.model_executor.runner.base_cuda_graph_runner import (
            get_batch_sizes_to_capture,
        )
        from sglang.srt.runtime_context import create_context, use_context
        from sglang.srt.server_args import ServerArgs

        args = ServerArgs(
            model_path="dummy",
            device="cuda",
            tp_size=8,
            speculative_algorithm="DSPARK",
            speculative_num_draft_tokens=6,
            max_running_requests=4,
            cuda_graph_bs_decode=[1, 2, 16],
            cuda_graph_backend_prefill="disabled",
            mem_fraction_static=0.8,
            chunked_prefill_size=2048,
        )
        parse_cuda_graph_config(args)
        with (
            patch(
                "sglang.srt.arg_groups.memory_hook.get_device_memory_capacity",
                return_value=288 * 1024,
            ),
            patch(
                "sglang.srt.arg_groups.memory_hook.use_mla_backend", return_value=True
            ),
        ):
            handle_gpu_memory_settings(args)
        # Resolve the relevant hooks without downloading a model or allocating
        # device storage. Config access, pool sizing and bucket selection are real.
        args._resolution_finished = True
        with (
            use_context(create_context(args, role="test")),
            patch.object(mb, "_gather_tp", side_effect=lambda value, **kw: [value] * 8),
        ):
            selected = mb._kimi_build_widths(64, 6, ARCH, is_draft=False)
            self.assertEqual(selected, (6, 12, 18, 24))
            sizing = object.__new__(KVCacheConfigurator)
            sizing.model_config = NS(context_len=262144)
            sizing.attn_dp_size = 1
            sizing.mambaish_config = None
            for capacity, expected_pool in ((2, 1), (6, 3), (10000, 4)):
                with self.subTest(capacity=capacity):
                    pool_size = sizing.resolve_max_num_reqs(capacity)
                    self.assertEqual(pool_size, expected_pool)
                    runner = NS(server_args=args, req_to_token_pool=NS(size=pool_size))
                    captures, _ = get_batch_sizes_to_capture(
                        runner, 6, gathered_buffer_required=False
                    )
                    self.assertEqual(max(captures), pool_size)
                    self.assertTrue({6 * bs for bs in captures} <= set(selected))
                    # The same bound covers specialized eager batches, too.
                    self.assertTrue(
                        {6 * bs for bs in range(1, pool_size + 1)} <= set(selected)
                    )
            self.assertEqual(
                mb.get_exec().graph.cuda_graph_config.decode.bs, [1, 2, 16]
            )


if __name__ == "__main__":
    unittest.main()
