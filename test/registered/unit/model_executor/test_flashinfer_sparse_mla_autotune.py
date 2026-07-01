import contextlib
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch

from sglang.srt.environ import envs
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.model_executor.runner.base_runner import BaseRunner
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class _TestRunner(BaseRunner):
    def can_run_graph(self, forward_batch):
        return False

    def load_batch(self, forward_batch, **kwargs):
        raise NotImplementedError

    def execute(self, forward_batch, **kwargs):
        raise NotImplementedError


class TestFlashInferSparseMLAAutotune(unittest.TestCase):
    def _runner(self, batch_size=512):
        runner = object.__new__(_TestRunner)
        server_args = SimpleNamespace(dsa_decode_backend="flashinfer_sparse_mla")
        model_runner = SimpleNamespace(
            server_args=server_args,
            forward_stream=MagicMock(),
            device="cuda",
            spec_algorithm=MagicMock(),
        )
        model_runner.spec_algorithm.is_speculative.return_value = False
        runner.model_runner = model_runner
        runner._dummy_run = MagicMock()
        runner._flashinfer_autotune_cache_path = MagicMock(
            return_value=Path("/tmp/sglang-test-flashinfer-autotune.json")
        )
        return runner, object(), batch_size

    def test_sparse_backend_enables_autotune(self):
        runner, _, _ = self._runner()
        runner.model_runner.server_args.disable_flashinfer_autotune = False
        runner.model_runner.server_args.moe_runner_backend = "triton"
        runner.model_runner.server_args.moe_a2a_backend = "none"
        runner.model_runner.model_config = SimpleNamespace(quantization=None)
        runner.model_runner.is_draft_worker = False
        disabled_backend = SimpleNamespace(
            is_flashinfer_cutlass=lambda: False,
            is_flashinfer_cutedsl=lambda: False,
        )

        with (
            patch(
                "sglang.srt.layers.quantization.fp4_utils.get_fp4_gemm_runner_backend",
                return_value=disabled_backend,
            ),
            patch(
                "sglang.srt.layers.quantization.fp8_utils.get_fp8_gemm_runner_backend",
                return_value=disabled_backend,
            ),
            patch("sglang.srt.utils.is_sm100_supported", return_value=False),
            patch("torch.cuda.get_device_capability", return_value=(12, 0)),
        ):
            self.assertTrue(runner._should_run_flashinfer_autotune())

            runner.model_runner.server_args.disable_flashinfer_autotune = True
            self.assertFalse(runner._should_run_flashinfer_autotune())

    def test_large_warmup_also_runs_decode_shape(self):
        runner, buffers, batch_size = self._runner()
        fake_device_module = SimpleNamespace(
            stream=lambda _stream: contextlib.nullcontext()
        )
        current_stream = MagicMock()

        with (
            envs.SGLANG_FLASHINFER_AUTOTUNE_CACHE.override(True),
            patch(
                "flashinfer.autotuner.autotune", return_value=contextlib.nullcontext()
            ),
            patch(
                "sglang.srt.layers.logits_processor.autotune_dummy_run_mode",
                return_value=contextlib.nullcontext(),
            ),
            patch("torch.cuda.current_stream", return_value=current_stream),
            patch("torch.get_device_module", return_value=fake_device_module),
        ):
            runner._flashinfer_autotune(buffers=buffers, batch_size=batch_size)

        self.assertEqual(
            runner._dummy_run.call_args_list,
            [
                call(batch_size=512, buffers=buffers),
                call(
                    batch_size=16,
                    buffers=buffers,
                    forward_mode_override=ForwardMode.DECODE,
                ),
            ],
        )

    def test_decode_sized_warmup_is_not_duplicated(self):
        runner, buffers, _ = self._runner(batch_size=16)
        fake_device_module = SimpleNamespace(
            stream=lambda _stream: contextlib.nullcontext()
        )
        current_stream = MagicMock()

        with (
            envs.SGLANG_FLASHINFER_AUTOTUNE_CACHE.override(True),
            patch(
                "flashinfer.autotuner.autotune", return_value=contextlib.nullcontext()
            ),
            patch(
                "sglang.srt.layers.logits_processor.autotune_dummy_run_mode",
                return_value=contextlib.nullcontext(),
            ),
            patch("torch.cuda.current_stream", return_value=current_stream),
            patch("torch.get_device_module", return_value=fake_device_module),
        ):
            runner._flashinfer_autotune(buffers=buffers, batch_size=16)

        runner._dummy_run.assert_called_once_with(batch_size=16, buffers=buffers)


if __name__ == "__main__":
    unittest.main()
