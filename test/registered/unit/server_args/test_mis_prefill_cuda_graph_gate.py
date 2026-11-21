"""Prefill/decode CUDA graph gate behavior for `--enable-mis`."""

import unittest
from unittest.mock import patch

import msgspec

import sglang.srt.arg_groups.attention_hook as attention_hook_module
from sglang.srt.arg_groups.attention_hook import handle_multi_item_scoring
from sglang.srt.arg_groups.memory_hook import handle_gpu_memory_settings
from sglang.srt.arg_groups.overrides import resolution_result
from sglang.srt.arg_groups.parallel_hook import handle_data_parallelism
from sglang.srt.model_executor.cuda_graph_config import (
    Backend,
    CudaGraphConfig,
    Phase,
    PhaseConfig,
)
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=6, suite="base-a-test-cpu")


class TestMISPrefillCudaGraphGate(CustomTestCase):
    def test_breakable_prefill_survives_while_decode_is_disabled(self):
        with self.assertLogs(attention_hook_module.logger, level="WARNING") as logs:
            cg = _run_mis_handler(
                enable_mis=True,
                prefill_backend=Backend.BREAKABLE,
                decode_backend=Backend.FULL,
            )

        self.assertEqual(cg.prefill.backend, Backend.BREAKABLE)
        self.assertEqual(cg.decode.backend, Backend.DISABLED)
        joined = "\n".join(logs.output)
        self.assertIn("Decode CUDA graph is disabled", joined)
        self.assertNotIn("Prefill CUDA graph backend", joined)

    def test_already_disabled_emits_no_cuda_graph_warning(self):
        with self.assertNoLogs(attention_hook_module.logger, level="WARNING"):
            cg = _run_mis_handler(
                enable_mis=True,
                prefill_backend=Backend.DISABLED,
                decode_backend=Backend.DISABLED,
            )

        self.assertEqual(cg.prefill.backend, Backend.DISABLED)
        self.assertEqual(cg.decode.backend, Backend.DISABLED)

    def test_unsupported_prefill_backends_are_disabled_with_warning(self):
        for prefill_backend in (Backend.FULL, Backend.TC_PIECEWISE):
            with (
                self.subTest(prefill_backend=prefill_backend),
                self.assertLogs(attention_hook_module.logger, level="WARNING") as logs,
            ):
                cg = _run_mis_handler(
                    enable_mis=True,
                    prefill_backend=prefill_backend,
                    decode_backend=Backend.FULL,
                )
                self.assertEqual(cg.prefill.backend, Backend.DISABLED)
                self.assertEqual(cg.decode.backend, Backend.DISABLED)
                self.assertIn("Prefill CUDA graph backend", "\n".join(logs.output))

    def test_disabled_mis_is_a_no_op(self):
        with self.assertNoLogs(attention_hook_module.logger, level="WARNING"):
            cg = _run_mis_handler(
                enable_mis=False,
                prefill_backend=Backend.FULL,
                decode_backend=Backend.FULL,
            )

        self.assertEqual(cg.prefill.backend, Backend.FULL)
        self.assertEqual(cg.decode.backend, Backend.FULL)

    def test_unchunked_prefill_materializes_positive_capture_buckets(self):
        cases = (
            (16 * 1024, None, 2048),
            (None, None, 4096),
            (80 * 1024, None, 8192),
            (16 * 1024, 3072, 3072),
        )
        for gpu_mem_mb, explicit_max_bs, expected_max_bs in cases:
            with self.subTest(gpu_mem_mb=gpu_mem_mb, explicit_max_bs=explicit_max_bs):
                sa = ServerArgs(
                    model_path="dummy",
                    enable_mis=True,
                    disable_radix_cache=True,
                    chunked_prefill_size=-1,
                    mem_fraction_static=0.25,
                    cuda_graph_config=CudaGraphConfig(
                        decode=PhaseConfig(backend=Backend.DISABLED),
                        prefill=PhaseConfig(
                            backend=Backend.BREAKABLE,
                            max_bs=explicit_max_bs,
                        ),
                    ),
                )
                sa._cuda_graph_config_locked = (
                    {(Phase.PREFILL, "max_bs")}
                    if explicit_max_bs is not None
                    else set()
                )
                with (
                    patch(
                        "sglang.srt.arg_groups.memory_hook.use_mla_backend",
                        return_value=False,
                    ),
                    patch(
                        "sglang.srt.arg_groups.memory_hook.get_device_memory_capacity",
                        return_value=gpu_mem_mb,
                    ),
                ):
                    handle_gpu_memory_settings(sa)
                prefill = resolution_result(sa, "cuda_graph_config").prefill
                self.assertEqual(prefill.max_bs, expected_max_bs)
                self.assertTrue(prefill.bs)
                self.assertEqual(prefill.bs[-1], expected_max_bs)
                self.assertEqual(resolution_result(sa, "chunked_prefill_size"), -1)

    def test_dp_attention_preserves_unchunked_mis_and_scales_prefill_budget(self):
        sa = ServerArgs(
            model_path="dummy",
            tp_size=2,
            attn_dp_size=2,
            enable_mis=True,
            disable_radix_cache=True,
            chunked_prefill_size=-1,
            cuda_graph_config=CudaGraphConfig(
                decode=PhaseConfig(backend=Backend.DISABLED),
                prefill=PhaseConfig(backend=Backend.BREAKABLE, max_bs=4096),
            ),
        )
        sa._cuda_graph_config_locked = set()
        handle_data_parallelism(sa)
        self.assertEqual(resolution_result(sa, "chunked_prefill_size"), -1)
        prefill = resolution_result(sa, "cuda_graph_config").prefill
        self.assertEqual(prefill.max_bs, 2048)
        self.assertTrue(prefill.bs)
        self.assertEqual(prefill.bs[-1], 2048)


def _run_mis_handler(*, enable_mis: bool, prefill_backend: str, decode_backend: str):
    sa = ServerArgs(
        model_path="dummy",
        enable_mis=enable_mis,
        attention_backend="flashinfer",
        disable_radix_cache=True,
        chunked_prefill_size=-1,
    )
    cg = CudaGraphConfig(
        prefill=PhaseConfig(backend=prefill_backend),
        decode=PhaseConfig(backend=decode_backend),
    )
    msgspec.Struct.__setattr__(sa, "cuda_graph_config", cg)
    handle_multi_item_scoring(sa)
    return resolution_result(sa, "cuda_graph_config")


if __name__ == "__main__":
    unittest.main()
