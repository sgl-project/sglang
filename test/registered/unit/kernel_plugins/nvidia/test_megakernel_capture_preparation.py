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
            patch.object(
                mb, "_gather_tp", side_effect=lambda value, **kw: [value] * 8
            ) as gather,
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
            for limit in (None, 0, 48):
                with (
                    self.subTest(limit=limit),
                    patch.object(
                        mb, "get_schedule", return_value=NS(max_running_requests=limit)
                    ),
                ):
                    self.assertEqual(
                        mb._kimi_build_widths(64, 6, ARCH, is_draft=False), DEFAULT
                    )
            with patch(
                "sglang.srt.layers.dp_attention.get_cuda_graph_batch_size_alignment",
                return_value=8,
            ):
                self.assertEqual(
                    mb._kimi_build_widths(64, 6, ARCH, is_draft=False), DEFAULT
                )
            gather.side_effect = lambda value, **kw: [value] * 7 + [DEFAULT]
            with self.assertRaisesRegex(RuntimeError, "ranks.*differ"):
                mb._kimi_build_widths(64, 6, ARCH, is_draft=False)


if __name__ == "__main__":
    unittest.main()
